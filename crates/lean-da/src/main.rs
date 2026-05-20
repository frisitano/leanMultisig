mod cache;
mod lean_air;

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
    #[arg(long, alias = "direct-air-census", help = "Print the LeanAIR table census and exit")]
    lean_air_census: bool,
    #[arg(
        long,
        alias = "direct-air-sweep",
        help = "Print a fused LeanAIR shape sweep and exit"
    )]
    lean_air_sweep: bool,
    #[arg(long, help = "Maximum blob count for --lean-air-sweep", default_value_t = 128)]
    max_n_blobs: usize,
    #[arg(
        long,
        help = "Cell size in extension-field elements for --lean-air-census",
        default_value_t = lean_air::DEFAULT_CELL_LEN_EXT
    )]
    cell_len_ext: usize,
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
    if cli.lean_air_census {
        print_lean_air_census(cli.n_blobs, cli.cell_len_ext);
        return;
    }
    if cli.lean_air_sweep {
        print_lean_air_sweep(cli.max_n_blobs);
        return;
    }

    let bytecode = compile_lean_da_bytecode(cli.n_blobs, cli.construction);
    let (witness, public_input) = build_instance(cli.n_blobs, cli.construction);
    let proof = prove_lean_da(&bytecode, &public_input, &witness, cli.n_blobs, cli.construction);
    verify_lean_da(&bytecode, &public_input, proof.proof);
}

fn data_kib(shape: lean_air::LeanAirShape) -> f64 {
    const F_BITS: usize = 31;
    (shape.data_len_base() * F_BITS) as f64 / (8.0 * 1024.0)
}

fn print_lean_air_census(n_blobs: usize, cell_len_ext: usize) {
    let shape = lean_air::LeanAirShape {
        log_m: LOG_M,
        n_rows: n_blobs,
        cell_len_ext,
    };
    let census = lean_air::estimate_air_census(shape);
    let plan = lean_air::LeanAirTablePlan::new(shape);
    let labels = [
        "CellHash",
        "RowDigest",
        "RowRoot",
        "ColumnMerkle",
        "ColumnRoot",
        "FinalRoot",
    ];

    println!("LeanAIR census");
    println!("n_rows:              {n_blobs}");
    println!("cell_len_ext:        {cell_len_ext}");
    println!("num_cells:           {}", shape.num_cells());
    println!("num_systematic_cells:{}", shape.num_systematic_cells());
    println!("Poseidon calls by kind:");
    for (label, count) in labels.iter().zip(census.poseidon_calls_by_kind) {
        println!("  {label:13} {count:>10}");
    }
    println!(
        "Poseidon rows:       {} -> padded {}",
        census.poseidon_rows, census.poseidon_rows_padded
    );
    println!(
        "Control rows:        {} -> padded {}",
        census.control_rows, census.control_rows_padded
    );
    println!(
        "Row parity rows:     {} -> padded {}",
        census.row_parity_rows, census.row_parity_rows_padded
    );
    println!("Total active rows:   {}", census.total_active_table_rows());
    println!("Total padded rows:   {}", census.total_padded_table_rows());
    println!(
        "Padding ratio:       {:.2}x",
        census.padding_overhead_bps() as f64 / 10_000.0
    );
    println!(
        "Fused total rows:    {} -> padded {}",
        census.fused_hash_active_table_rows(),
        census.fused_hash_padded_table_rows()
    );
    println!(
        "Fused row reduction: {:.2}%",
        census.fused_hash_row_reduction_bps() as f64 / 100.0
    );
    println!(
        "RLC parity rows:     {} -> padded {}",
        census.materialized_rlc_parity_rows(),
        census.materialized_rlc_parity_rows_padded()
    );
    println!(
        "Fused + RLC rows:    {} -> padded {}",
        census.fused_hash_materialized_rlc_rows(),
        census.fused_hash_materialized_rlc_padded_rows()
    );
    println!(
        "Fused virtual RLC:   padded {}",
        census.fused_hash_virtual_rlc_padded_rows()
    );
    println!("Cell digests:        {}", census.cell_digest_count());
    println!("Digest bindings:     {}", census.digest_binding_entries());
    println!("Duplicated memory:   {} FE", census.duplicated_memory_fes);
    println!("Shared memory est.:  {} FE", census.shared_memory_fes_estimate);
    println!("Hash chunks/cell:    {}", plan.hash.chunks_per_cell);
    println!("Hash schedule:");
    print_row_range("  cell_hash", plan.hash.cell_hash);
    print_row_range("  row_digest", plan.hash.row_digest);
    print_row_range("  row_root", plan.hash.row_root);
    print_row_range("  column_merkle", plan.hash.column_merkle);
    print_row_range("  column_root", plan.hash.column_root);
    print_row_range("  final_root", plan.hash.final_root);
    println!(
        "RowParity schedule:  rows {}..{} -> padded {}",
        0, plan.row_parity.active_rows, plan.row_parity.padded_rows
    );
}

fn print_row_range(label: &str, range: lean_air::RowRange) {
    println!("{label:16} {}..{} ({})", range.start, range.end(), range.len);
}

fn print_lean_air_sweep(max_n_blobs: usize) {
    let mut candidates = Vec::new();
    for cell_len_ext in [16, 32, 64, 128] {
        for n_rows in 1..=max_n_blobs {
            let shape = lean_air::LeanAirShape {
                log_m: LOG_M,
                n_rows,
                cell_len_ext,
            };
            if !shape.codeword_len_ext().is_multiple_of(cell_len_ext) {
                continue;
            }
            let census = lean_air::estimate_air_census(shape);
            let kib = data_kib(shape);
            let fused_rows = census.fused_hash_padded_table_rows();
            let kib_per_mrow = kib * 1_000_000.0 / fused_rows as f64;
            candidates.push((kib_per_mrow, kib, census));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!("Top fused LeanAIR shapes by data per padded table row");
    println!(
        "{:>4} {:>5} {:>5} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "rank", "rows", "cell", "data KiB", "hash", "hash pad", "ldt pad", "pad", "KiB/Mrow"
    );
    for (rank, (kib_per_mrow, kib, census)) in candidates.into_iter().take(24).enumerate() {
        println!(
            "{:>4} {:>5} {:>5} {:>10.1} {:>10} {:>10} {:>10} {:>7.2}x {:>8.1}",
            rank + 1,
            census.shape.n_rows,
            census.shape.cell_len_ext,
            kib,
            census.poseidon_rows,
            census.poseidon_rows_padded,
            census.row_parity_rows_padded,
            census.fused_hash_padding_overhead_bps() as f64 / 10_000.0,
            kib_per_mrow,
        );
    }
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
    let num_systematic_leaves = (1 << LOG_M) / leaf_len_ext;

    let rows: Vec<Vec<F>> = codewords
        .iter()
        .map(|codeword| serialize_codeword_for_witness(codeword))
        .collect();

    let mut leaf_digests = vec![[F::ZERO; 8]; num_leaves * n_blobs_padded];
    for (row_idx, row) in rows.iter().enumerate() {
        for leaf_idx in 0..num_leaves {
            let start = leaf_idx * leaf_len;
            let leaf = &row[start..start + leaf_len];
            leaf_digests[leaf_idx * n_blobs_padded + row_idx] = utils::poseidon_compress_slice(leaf, false);
        }
    }

    let mut row_digests = Vec::with_capacity(n_blobs);
    for row_idx in 0..n_blobs {
        let systematic_digests =
            (0..num_systematic_leaves).map(|leaf_idx| leaf_digests[leaf_idx * n_blobs_padded + row_idx]);
        row_digests.push(chain_hash_digests(systematic_digests));
    }
    let row_commitment_root = chain_hash_digests(row_digests);

    let mut column_roots = Vec::with_capacity(num_leaves);
    for leaf_idx in 0..num_leaves {
        let start = leaf_idx * n_blobs_padded;
        let cell_digests = leaf_digests[start..start + n_blobs_padded].to_vec();
        column_roots.push(merkle_root_from_digests(cell_digests));
    }

    let column_commitment_root = merkle_root_from_digests(column_roots);
    let commitment_root = utils::poseidon16_compress_pair(&row_commitment_root, &column_commitment_root);
    commitment_root
}

fn chain_hash_digests(digests: impl IntoIterator<Item = [F; 8]>) -> [F; 8] {
    let mut state = [F::ZERO; 8];
    for digest in digests {
        state = utils::poseidon16_compress_pair(&state, &digest);
    }
    state
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
    fn test_lean_air_commitment_matches_column_commit_public_input() {
        let n_blobs = 3;
        let codewords = generate_codewords(n_blobs);
        let expected = build_column_commit_public_input(&codewords);
        let shape = lean_air::LeanAirShape::new(LOG_M, n_blobs);
        let trace = lean_air::build_trace_evens_then_odds(shape, &codewords);
        let commitments = lean_air::commit_codewords_evens_then_odds(shape, &codewords);

        assert_eq!(commitments.commitment_root, expected);
        assert_eq!(commitments.row_digests.len(), n_blobs);
        assert_eq!(commitments.column_roots.len(), shape.num_cells());
        assert_eq!(trace.commitments, commitments);
        assert!(!trace.poseidon_calls.is_empty());
    }

    #[test]
    fn test_lean_air_precompile_layout_matches_poseidon_events() {
        let n_blobs = 3;
        let codewords = generate_codewords(n_blobs);
        let shape = lean_air::LeanAirShape::new(LOG_M, n_blobs);
        let trace = lean_air::build_trace_evens_then_odds(shape, &codewords);
        let layout = lean_air::build_precompile_layout(&trace);

        assert_eq!(layout.poseidon_requests.len(), trace.poseidon_calls.len());
        for (request, call) in layout.poseidon_requests.iter().zip(&trace.poseidon_calls) {
            assert_eq!(layout.memory[request.left_ptr..request.left_ptr + 8], call.input[..8]);
            assert_eq!(layout.memory[request.right_ptr..request.right_ptr + 8], call.input[8..]);
            assert_eq!(layout.memory[request.result_ptr..request.result_ptr + 8], call.output);
        }
    }

    #[test]
    fn test_lean_air_census_is_consistent() {
        let n_blobs = 3;
        let codewords = generate_codewords(n_blobs);
        let shape = lean_air::LeanAirShape::new(LOG_M, n_blobs);
        let trace = lean_air::build_trace_evens_then_odds(shape, &codewords);
        let census = lean_air::air_census(shape, &trace);
        let estimate = lean_air::estimate_air_census(shape);

        assert_eq!(census.poseidon_calls(), trace.poseidon_calls.len());
        assert_eq!(census, estimate);
        assert_eq!(
            census.poseidon_calls_by_kind.iter().sum::<usize>(),
            trace.poseidon_calls.len()
        );
        assert_eq!(census.control_rows, trace.poseidon_calls.len());
        assert_eq!(census.row_parity_rows, n_blobs * (1 << LOG_M));
        assert_eq!(census.materialized_rlc_parity_rows(), 1 << LOG_M);
        assert!(census.fused_hash_materialized_rlc_padded_rows() < census.fused_hash_padded_table_rows());
        assert!(census.fused_hash_virtual_rlc_padded_rows() < census.fused_hash_materialized_rlc_padded_rows());
        assert!(census.fused_hash_padded_table_rows() < census.total_padded_table_rows());
        assert!(census.shared_memory_fes_estimate < census.duplicated_memory_fes);
    }

    #[test]
    fn test_lean_air_schedules_are_consistent() {
        let n_blobs = 3;
        let shape = lean_air::LeanAirShape {
            log_m: LOG_M,
            n_rows: n_blobs,
            cell_len_ext: 64,
        };
        let plan = lean_air::LeanAirTablePlan::new(shape);
        let census = lean_air::estimate_air_census(shape);

        assert_eq!(plan.hash.active_rows(), census.poseidon_rows);
        assert_eq!(plan.hash.padded_rows, census.poseidon_rows_padded);
        assert_eq!(plan.row_parity.active_rows, census.row_parity_rows);
        assert_eq!(plan.row_parity.padded_rows, census.row_parity_rows_padded);

        let first_hash = plan.hash.row_meta(0).unwrap();
        assert_eq!(first_hash.kind, lean_air::PoseidonCallKind::CellHash);
        assert_eq!(first_hash.row_idx, Some(0));
        assert_eq!(first_hash.cell_idx, Some(0));
        assert_eq!(first_hash.chunk_idx, Some(0));

        let last_hash = plan.hash.row_meta(plan.hash.active_rows() - 1).unwrap();
        assert_eq!(last_hash.kind, lean_air::PoseidonCallKind::FinalRoot);
        assert_eq!(plan.hash.row_meta(plan.hash.active_rows()), None);

        let first_parity = plan.row_parity.row_meta(0).unwrap();
        assert_eq!(first_parity.row_idx, 0);
        assert_eq!(first_parity.term_idx, 0);
        assert!(first_parity.is_first);
        assert!(!first_parity.is_last);

        let last_parity = plan.row_parity.row_meta(plan.row_parity.active_rows - 1).unwrap();
        assert_eq!(last_parity.row_idx, n_blobs - 1);
        assert_eq!(last_parity.term_idx, shape.message_len_ext() - 1);
        assert!(!last_parity.is_first);
        assert!(last_parity.is_last);
        assert_eq!(plan.row_parity.row_meta(plan.row_parity.active_rows), None);
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
