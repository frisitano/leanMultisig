mod cache;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use backend::{Algebra, BasedVectorSpace, PrimeCharacteristicRing, Proof, TwoAdicField};
use clap::{Parser, ValueEnum};
use lean_compiler::{CompilationFlags, ProgramSource, compile_program_with_flags};
use lean_prover::{
    default_whir_config,
    prove_execution::{ExecutionProof, prove_execution},
    verify_execution::verify_execution,
};
use lean_vm::{Bytecode, EF, ExecutionWitness, F};
use rand::{RngExt, SeedableRng, rngs::StdRng};

static EMBEDDED_ZK_DSL: include_dir::Dir<'_> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/zkdsl_implem");

const STARTING_LOG_INV_RATE: usize = 1;

pub const LOG_M: usize = 13; // Blob ≈ 155 KiB = 2^13 extension field elements (= 2^13 * 5 base field elements)
pub const DEFAULT_N_BLOBS: usize = 8;

#[derive(Parser)]
#[command(about = "Reed-Solomon DA: hash N_BLOBS codewords, run a random parity check, then prove + verify")]
struct Cli {
    #[arg(long, help = "Number of blobs to commit", default_value_t = DEFAULT_N_BLOBS)]
    n_blobs: usize,
    #[arg(long, value_enum, default_value_t = Construction::Baseline, help = "leanDA construction to run")]
    construction: Construction,
    #[arg(long, help = "Enable tracing")]
    tracing: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Construction {
    Baseline,
    ColumnCommit,
}

impl Construction {
    fn entry(self) -> &'static str {
        match self {
            Self::Baseline => "lean_da.py",
            Self::ColumnCommit => "lean_da_column_commit.py",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ColumnCommit => "column-commit",
        }
    }
}

fn main() {
    // cargo run --release -p lean-da -- --n-blobs 48
    let cli = Cli::parse();
    if cli.tracing {
        utils::init_tracing();
    }

    let bytecode = compile_lean_da_bytecode(cli.n_blobs, cli.construction);
    let (witness, public_input) = build_instance(cli.n_blobs, cli.construction);
    let proof = prove_lean_da(&bytecode, &public_input, &witness, cli.n_blobs, cli.construction);
    verify_lean_da(&bytecode, &public_input, proof.proof);
}

pub fn compile_lean_da_bytecode(n_blobs: usize, construction: Construction) -> Bytecode {
    assert!(n_blobs > 0, "n_blobs must be nonzero");
    let n_blobs_padded = n_blobs.next_power_of_two();

    let mut replacements = BTreeMap::new();
    replacements.insert("LEAN_DA_ENTRY".to_string(), construction.entry().to_string());
    replacements.insert("LOG_M_PLACEHOLDER".to_string(), LOG_M.to_string());
    replacements.insert("N_BLOBS_PLACEHOLDER".to_string(), n_blobs.to_string());
    replacements.insert("N_BLOBS_PADDED_PLACEHOLDER".to_string(), n_blobs_padded.to_string());
    replacements.insert(
        "LOG_N_BLOBS_PADDED_PLACEHOLDER".to_string(),
        log2_exact_power_of_two(n_blobs_padded).to_string(),
    );

    if let Some((bytecode, _)) = cache::try_load(&EMBEDDED_ZK_DSL, &replacements) {
        println!("(Compilation cache hit)");
        return bytecode;
    }

    let time = Instant::now();
    let source = ProgramSource::Embedded {
        entry: construction.entry().to_string(),
        dir: &EMBEDDED_ZK_DSL,
    };
    let bytecode = compile_program_with_flags(
        &source,
        CompilationFlags {
            replacements: replacements.clone(),
        },
    );
    println!("Compilation time: {:.3} s", time.elapsed().as_secs_f64());

    if let Err(e) = cache::try_store(&EMBEDDED_ZK_DSL, &replacements, &bytecode) {
        eprintln!("Warning: failed to write bytecode cache: {e}");
    }

    bytecode
}

fn ntt<A: Algebra<F> + Copy>(a: &mut [A]) {
    let n = a.len();
    assert!(n.is_power_of_two());
    let log_n = n.trailing_zeros() as usize;

    let shift = usize::BITS as usize - log_n;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            a.swap(i, j);
        }
    }

    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let root = F::two_adic_generator(size.trailing_zeros() as usize);
        for chunk_start in (0..n).step_by(size) {
            let mut twiddle = F::ONE;
            for i in 0..half {
                let u = a[chunk_start + i];
                let v = a[chunk_start + i + half] * twiddle;
                a[chunk_start + i] = u + v;
                a[chunk_start + i + half] = u - v;
                twiddle *= root;
            }
        }
        size *= 2;
    }
}

fn rs_encode<A: Algebra<F> + Copy>(message: &[A]) -> Vec<A> {
    let m = message.len();
    assert!(m.is_power_of_two());
    let mut codeword = vec![A::ZERO; 2 * m];
    codeword[..m].copy_from_slice(message);
    ntt(&mut codeword);
    codeword
}

fn log2_exact_power_of_two(n: usize) -> usize {
    assert!(n > 0 && n.is_power_of_two());
    n.trailing_zeros() as usize
}

fn generate_codewords(n_blobs: usize) -> Vec<Vec<EF>> {
    let m = 1 << LOG_M;
    let mut rng = StdRng::seed_from_u64(0);
    let mut codewords = Vec::with_capacity(n_blobs);
    for _ in 0..n_blobs {
        let message: Vec<EF> = (0..m).map(|_| rng.random()).collect();
        codewords.push(rs_encode(&message));
    }
    codewords
}

fn push_ext(out: &mut Vec<F>, value: &EF) {
    out.extend_from_slice(<EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value));
}

fn serialize_codeword_for_witness(codeword: &[EF]) -> Vec<F> {
    let m = 1 << LOG_M;
    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    let mut codeword_f: Vec<F> = Vec::with_capacity(2 * m * dim);
    for j in 0..m {
        push_ext(&mut codeword_f, &codeword[2 * j]);
    }
    for j in 0..m {
        push_ext(&mut codeword_f, &codeword[2 * j + 1]);
    }
    codeword_f
}

fn build_witness_from_codewords(codewords: &[Vec<EF>]) -> ExecutionWitness {
    let mut hints: HashMap<String, Vec<Vec<F>>> = HashMap::new();
    let codeword_blobs = codewords
        .iter()
        .map(|codeword| serialize_codeword_for_witness(codeword))
        .collect();
    hints.insert("codeword".to_string(), codeword_blobs);
    ExecutionWitness {
        preamble_memory_len: 0,
        hints,
    }
}

fn build_instance(n_blobs: usize, construction: Construction) -> (ExecutionWitness, Vec<F>) {
    let codewords = generate_codewords(n_blobs);
    let public_input = match construction {
        Construction::Baseline => vec![],
        Construction::ColumnCommit => build_column_commit_public_input(&codewords).to_vec(),
    };
    let witness = build_witness_from_codewords(&codewords);
    (witness, public_input)
}

fn build_column_commit_public_input(codewords: &[Vec<EF>]) -> [F; 8] {
    let n_blobs = codewords.len();
    assert!(n_blobs > 0);
    let n_blobs_padded = n_blobs.next_power_of_two();

    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    let leaf_len_ext = 1 << 4;
    let leaf_len = leaf_len_ext * dim;
    let row_len = 2 * (1 << LOG_M);
    let num_leaves = row_len / leaf_len_ext;

    let rows: Vec<Vec<F>> = codewords
        .iter()
        .map(|codeword| serialize_codeword_for_witness(codeword))
        .collect();

    let mut column_roots = Vec::with_capacity(num_leaves);
    for leaf_idx in 0..num_leaves {
        let mut cell_digests = Vec::with_capacity(n_blobs);
        for row in &rows {
            let start = leaf_idx * leaf_len;
            let leaf = &row[start..start + leaf_len];
            cell_digests.push(utils::poseidon_compress_slice(leaf, false));
        }
        for _ in n_blobs..n_blobs_padded {
            cell_digests.push([F::ZERO; 8]);
        }
        column_roots.push(merkle_root_from_digests(cell_digests));
    }

    merkle_root_from_digests(column_roots)
}

fn merkle_root_from_digests(mut layer: Vec<[F; 8]>) -> [F; 8] {
    assert!(!layer.is_empty());
    assert!(layer.len().is_power_of_two());

    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|pair| utils::poseidon16_compress_pair(&pair[0], &pair[1]))
            .collect();
    }
    layer[0]
}

pub fn prove_lean_da(
    bytecode: &Bytecode,
    public_input: &[F],
    witness: &ExecutionWitness,
    n_blobs: usize,
    construction: Construction,
) -> ExecutionProof {
    const F_BITS: usize = 31;

    let t0 = Instant::now();
    let proof = prove_execution(
        bytecode,
        public_input,
        witness,
        &default_whir_config(STARTING_LOG_INV_RATE),
        false,
    )
    .unwrap();
    let proving_time = t0.elapsed();

    let meta = proof.metadata.as_ref().expect("metadata missing");
    let proof_size_fe = proof.proof.proof_size_fe();
    let proof_kib = (proof_size_fe * F_BITS) as f64 / (8.0 * 1024.0);
    // Each blob is 2^LOG_M extension elements, i.e. 2^LOG_M * DIM base field elements.
    let blob_size_fe = (1 << LOG_M) * <EF as BasedVectorSpace<F>>::DIMENSION;
    let total_data_kib = (n_blobs * blob_size_fe * F_BITS) as f64 / (8.0 * 1024.0);
    let throughput_kib_per_s = total_data_kib / proving_time.as_secs_f64();
    println!("Construction:     {}", construction.label());
    println!("Bytecode size:    {}", meta.bytecode_size);
    println!("Cycles:           {}", meta.cycles);
    println!("Poseidon16 calls: {}", meta.n_poseidons);
    println!("ExtensionOp calls:{}", meta.n_extension_ops);
    println!("Proving time:     {:.3} s", proving_time.as_secs_f64());
    println!("Proof size:       {:.2} KiB", proof_kib);
    println!(
        "Throughput:       {:.2} KiB/s ({} blobs * {} FE / {:.3} s)",
        throughput_kib_per_s,
        n_blobs,
        blob_size_fe,
        proving_time.as_secs_f64()
    );

    proof
}

pub fn verify_lean_da(bytecode: &Bytecode, public_input: &[F], proof: Proof<F>) {
    verify_execution(bytecode, public_input, proof).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_prove_verify() {
        // cargo test --release --package lean-da -- tests::test_compile_prove_verify --nocapture
        let bytecode = compile_lean_da_bytecode(DEFAULT_N_BLOBS, Construction::Baseline);
        let (witness, public_input) = build_instance(DEFAULT_N_BLOBS, Construction::Baseline);
        let proof = prove_lean_da(
            &bytecode,
            &public_input,
            &witness,
            DEFAULT_N_BLOBS,
            Construction::Baseline,
        );
        verify_lean_da(&bytecode, &public_input, proof.proof);
    }

    #[test]
    fn test_column_commit_public_input_shape() {
        let (_witness, public_input) = build_instance(DEFAULT_N_BLOBS, Construction::ColumnCommit);
        assert_eq!(public_input.len(), 8);
    }

    #[test]
    fn test_rs_encode_matches_naive() {
        let mut rng = StdRng::seed_from_u64(7);
        let m: usize = 1 << LOG_M;
        let message: Vec<EF> = (0..m).map(|_| rng.random()).collect();
        let two_m = 2 * m;
        let w = F::two_adic_generator(two_m.trailing_zeros() as usize);
        let naive: Vec<EF> = (0..two_m)
            .map(|j| {
                let wj = w.exp_u64(j as u64);
                message.iter().rev().fold(EF::ZERO, |acc, &c| acc * wj + c)
            })
            .collect();
        assert_eq!(rs_encode(&message), naive);
    }
}
