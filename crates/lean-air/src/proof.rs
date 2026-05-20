use backend::{
    Air, Algebra, BasedVectorSpace, ChallengeSampler, ConstraintFolderPacked, DensePolynomial, EFPacking, FSProver,
    FSVerifier, Field, FoldingFactor, MleGroup, MleGroupOwned, MleGroupRef, MleOwned, MultilinearPoint, PF, PFPacking,
    PackedFieldExtension, PackedValue, PrimeCharacteristicRing, Proof, ProofError, SparseStatement, SparseValue,
    SplitEq, SumcheckComputation, TwoAdicField, VerifierState, WhirConfig, WhirConfigBuilder, eval_eq,
    lagrange_basis_evals, mle_of_zeros_then_ones, packing_log_width, packing_width, sumcheck_verify, uninitialized_vec,
};
use lean_prover::default_whir_config;
use lean_vm::{DIGEST_LEN, EF, ExtraDataForBuses, F};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, Sub};
use sub_protocols::{OuterSumcheckSession, natural_ordering_point_for_session, prove_batched_air_sumcheck};
use tracing::info_span;
use utils::{build_prover_state, get_poseidon16};

use crate::{
    DEDICATED_HASH_INPUT_START, DEDICATED_HASH_NUM_COLS, DEDICATED_HASH_OUTPUT_START, DEDICATED_HASH_WIDTH,
    LeanAirDedicatedHashAir, LeanAirDedicatedHashTable, LeanAirShape,
};

#[derive(Clone, Debug)]
pub struct LeanAirProof {
    pub proof: Proof<F>,
    pub metadata: LeanAirProofMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanAirProofMetadata {
    pub shape: LeanAirShape,
    pub commitment_root: [F; DIGEST_LEN],
    pub global_n_vars: usize,
    pub poseidon_log_rows: usize,
    pub poseidon_columns: usize,
    pub proof_size_fe: usize,
}

pub fn prove_lean_air(shape: LeanAirShape, codewords: &[Vec<EF>]) -> LeanAirProof {
    prove_lean_air_with_config(shape, codewords, &default_lean_air_whir_config(shape))
}

pub fn prove_lean_air_with_config(
    shape: LeanAirShape,
    codewords: &[Vec<EF>],
    whir_config_builder: &WhirConfigBuilder,
) -> LeanAirProof {
    let table = LeanAirDedicatedHashTable::build(shape, codewords);
    prove_lean_air_table_with_config(table, whir_config_builder)
}

pub fn prove_lean_air_table(table: LeanAirDedicatedHashTable) -> LeanAirProof {
    let config = default_lean_air_whir_config(table.shape);
    prove_lean_air_table_with_config(table, &config)
}

pub fn prove_lean_air_table_with_config(
    table: LeanAirDedicatedHashTable,
    whir_config_builder: &WhirConfigBuilder,
) -> LeanAirProof {
    let shape = table.shape;
    let public = public_scalars(&table);
    let global = {
        let _cpu = system_info::enter_cpu_stage("prove/global_polynomial");
        info_span!("lean-air global polynomial").in_scope(|| global_polynomial(&table))
    };
    let global_n_vars = global.by_ref().n_vars();
    let actual_data_len = table.committed_columns().len() * table.padded_rows();
    let whir = WhirConfig::new(whir_config_builder, global_n_vars);

    let mut prover_state = build_prover_state();
    prover_state.observe_scalars(&public);
    let witness = {
        let _cpu = system_info::enter_cpu_stage("prove/whir/commit");
        info_span!("lean-air WHIR commit").in_scope(|| whir.commit(&mut prover_state, &global, actual_data_len))
    };

    let poseidon_log_rows = log2_strict(table.padded_rows());
    let eq_point = prover_state.sample_vec(poseidon_log_rows);
    let air_alpha = prover_state.sample();
    let row_coeff_alpha = row_coefficient_alpha(shape, table.commitments.commitment_root);
    let parity_point = prover_state.sample();
    let cell_link_point = prover_state.sample();
    let link_point = prover_state.sample();
    let link_mix = prover_state.sample();
    let eta = prover_state.sample();
    let extra_data = extra_data_for_air(air_alpha);

    debug_assert_eq!(
        weighted_air_sum(&table, &eq_point, &extra_data),
        EF::ZERO,
        "honest Poseidon table should satisfy the weighted AIR claim"
    );

    let parity_coefficients = {
        let _cpu = system_info::enter_cpu_stage("prove/parity_coefficients");
        info_span!("lean-air parity sparse coefficients")
            .in_scope(|| parity_sparse_coefficients(shape, table.padded_rows(), row_coeff_alpha, parity_point))
    };
    let cell_link_columns = cell_link_column_indexes();
    let cell_link_coefficients = {
        let _cpu = system_info::enter_cpu_stage("prove/cell_link_coefficients");
        info_span!("lean-air cell-chain dense coefficients")
            .in_scope(|| cell_link_dense_coefficients(shape, table.padded_rows(), cell_link_point))
    };
    let link_columns = link_column_indexes();
    let link_coefficients = {
        let _cpu = system_info::enter_cpu_stage("prove/link_coefficients");
        info_span!("lean-air link sparse coefficients")
            .in_scope(|| link_sparse_coefficients(shape, table.padded_rows(), link_point))
    };
    let linear_coefficients = {
        let _cpu = system_info::enter_cpu_stage("prove/mix_coefficients");
        info_span!("lean-air mix sparse coefficients")
            .in_scope(|| mixed_sparse_linear_coefficients(parity_coefficients, link_coefficients, link_mix))
    };
    debug_assert_eq!(
        dense_linear_sum(&table, &cell_link_columns, &cell_link_coefficients),
        EF::ZERO,
        "honest incremental cell hash links must satisfy the dense linear claim"
    );
    debug_assert_eq!(
        sparse_linear_sum(&table, &link_columns, &linear_coefficients),
        EF::ZERO,
        "honest codeword parity and non-cell hash links must satisfy the virtual linear-bus claim"
    );

    let column_refs = info_span!("lean-air collect column refs")
        .in_scope(|| table.committed_columns().iter().map(Vec::as_slice).collect::<Vec<_>>());
    let packed = info_span!("lean-air pack columns").in_scope(|| backend::MleGroupRef::<EF>::Base(column_refs).pack());
    let _session_cpu = system_info::enter_cpu_stage("prove/session_setup");
    let mut sessions: Vec<Box<dyn OuterSumcheckSession<EF>>> =
        vec![Box::new(info_span!("lean-air dedicated hash AIR session").in_scope(
            || DedicatedHashAirSumcheckSession::new(packed, eq_point.clone(), extra_data, table.active_rows()),
        ))];
    sessions.push(Box::new(info_span!("lean-air dense cell-chain session").in_scope(
        || {
            DenseLinearBusSession::new(
                table.committed_columns(),
                cell_link_columns,
                cell_link_coefficients,
                poseidon_log_rows,
            )
        },
    )));
    sessions.push(Box::new(info_span!("lean-air sparse linear session").in_scope(|| {
        VirtualLinearBusSession::new(
            table.committed_columns(),
            link_columns,
            linear_coefficients,
            poseidon_log_rows,
        )
    })));
    drop(_session_cpu);
    let sumcheck_air_point = info_span!("lean-air batched AIR+linear sumcheck").in_scope(|| {
        let _cpu = system_info::enter_cpu_stage("prove/air_linear_sumcheck");
        prove_batched_air_sumcheck(&mut prover_state, &mut sessions, eta)
    });
    let col_evals = sessions[0].final_column_evals();
    prover_state.add_extension_scalars(&col_evals);

    let statements = whir_statements(&table, global_n_vars, &sumcheck_air_point.0, &col_evals);
    {
        let _cpu = system_info::enter_cpu_stage("prove/whir/prove");
        info_span!("lean-air WHIR prove")
            .in_scope(|| whir.prove(&mut prover_state, statements, witness, &global.by_ref()));
    }

    let proof = prover_state.into_proof();
    let proof_size_fe = proof.proof_size_fe();
    LeanAirProof {
        proof,
        metadata: LeanAirProofMetadata {
            shape,
            commitment_root: table.commitments.commitment_root,
            global_n_vars,
            poseidon_log_rows,
            poseidon_columns: table.committed_columns().len(),
            proof_size_fe,
        },
    }
}

pub fn verify_lean_air(
    shape: LeanAirShape,
    commitment_root: [F; DIGEST_LEN],
    proof: Proof<F>,
) -> Result<LeanAirProofMetadata, ProofError> {
    verify_lean_air_with_config(shape, commitment_root, proof, &default_lean_air_whir_config(shape))
}

pub fn verify_lean_air_with_config(
    shape: LeanAirShape,
    commitment_root: [F; DIGEST_LEN],
    proof: Proof<F>,
    whir_config_builder: &WhirConfigBuilder,
) -> Result<LeanAirProofMetadata, ProofError> {
    let mut verifier_state = VerifierState::<EF, _>::new(proof, get_poseidon16().clone())?;

    let poseidon_rows = poseidon_rows_for_shape(shape);
    let poseidon_log_rows = log2_strict(poseidon_rows);
    let poseidon_columns = DEDICATED_HASH_NUM_COLS;
    let global_n_vars = log2_ceil(poseidon_columns * poseidon_rows);
    let public = public_scalars_from_shape(shape, commitment_root, poseidon_rows, poseidon_columns, global_n_vars);
    verifier_state.observe_scalars(&public);

    let whir = WhirConfig::new(whir_config_builder, global_n_vars);
    let parsed_commitment = whir.parse_commitment::<F>(&mut verifier_state)?;

    let eq_point = verifier_state.sample_vec(poseidon_log_rows);
    let air_alpha = verifier_state.sample();
    let row_coeff_alpha = row_coefficient_alpha(shape, commitment_root);
    let parity_point = verifier_state.sample();
    let cell_link_point = verifier_state.sample();
    let link_point = verifier_state.sample();
    let link_mix = verifier_state.sample();
    let eta = verifier_state.sample();
    let extra_data = extra_data_for_air(air_alpha);
    let air = LeanAirDedicatedHashAir;

    let backend::Evaluation {
        point: sumcheck_air_point,
        value: claimed_combined_final_value,
    } = sumcheck_verify(
        &mut verifier_state,
        poseidon_log_rows,
        (air.degree_air() + 1).max(VIRTUAL_LINEAR_BUS_BARE_DEGREE + 1),
        EF::ZERO,
        None,
    )?;

    let col_evals = verifier_state.next_extension_scalars_vec(poseidon_columns)?;
    let natural_point = natural_ordering_point_for_session(&sumcheck_air_point.0, poseidon_log_rows);
    let constraint_eval = air.eval_extension(&col_evals, &extra_data);
    let eq_val = MultilinearPoint(eq_point).eq_poly_outside(&MultilinearPoint(natural_point.clone()));
    let linear_coeff_evals = linear_coefficient_evals_from_schedule(
        shape,
        poseidon_rows,
        row_coeff_alpha,
        parity_point,
        link_point,
        link_mix,
        &sumcheck_air_point.0,
    );
    let cell_link_coeff_evals =
        cell_link_coefficient_evals_from_schedule(shape, poseidon_rows, cell_link_point, &sumcheck_air_point.0);
    let cell_link_eval = cell_link_coeff_evals
        .iter()
        .zip(cell_link_column_indexes())
        .map(|(&coeff, column)| coeff * col_evals[column])
        .sum::<EF>();
    let linear_eval = linear_coeff_evals
        .iter()
        .zip(link_column_indexes())
        .map(|(&coeff, column)| coeff * col_evals[column])
        .sum::<EF>();
    let linear_eq_val = constant_half_eq_eval(poseidon_log_rows);
    let expected_combined_final_value =
        eq_val * constraint_eval + eta * linear_eq_val * cell_link_eval + eta.square() * linear_eq_val * linear_eval;
    if claimed_combined_final_value != expected_combined_final_value {
        return Err(ProofError::InvalidProof);
    }

    let statements = whir_statements_from_public(
        shape,
        poseidon_rows,
        global_n_vars,
        &natural_point,
        &col_evals,
        commitment_root,
    );
    whir.verify(&mut verifier_state, &parsed_commitment, statements)?;

    let raw = verifier_state.into_raw_proof();
    let proof_size_fe = raw.transcript.len()
        + raw
            .merkle_openings
            .iter()
            .map(|opening| opening.leaf_data.len() + opening.path.len() * DIGEST_LEN)
            .sum::<usize>();

    Ok(LeanAirProofMetadata {
        shape,
        commitment_root,
        global_n_vars,
        poseidon_log_rows,
        poseidon_columns,
        proof_size_fe,
    })
}

fn weighted_air_sum(table: &LeanAirDedicatedHashTable, eq_point: &[EF], extra_data: &ExtraDataForBuses<EF>) -> EF {
    let eq_evals = backend::eval_eq(eq_point);
    assert_eq!(eq_evals.len(), table.padded_rows());
    let air = LeanAirDedicatedHashAir;
    let mut sum = EF::ZERO;
    for (row, eq_eval) in eq_evals.iter().copied().enumerate() {
        let point = table
            .committed_columns()
            .iter()
            .map(|column| EF::from(column[row]))
            .collect::<Vec<_>>();
        sum += eq_eval * air.eval_extension(&point, extra_data);
    }
    sum
}

fn global_polynomial(table: &LeanAirDedicatedHashTable) -> MleOwned<EF> {
    let actual_data_len = table.committed_columns().len() * table.padded_rows();
    let mut data = F::zero_vec(1 << log2_ceil(actual_data_len));
    data[..actual_data_len]
        .par_chunks_mut(table.padded_rows())
        .zip(table.committed_columns().par_iter())
        .for_each(|(dst, column)| dst.copy_from_slice(column));
    MleOwned::Base(data)
}

fn whir_statements(
    table: &LeanAirDedicatedHashTable,
    global_n_vars: usize,
    sumcheck_air_point: &[EF],
    col_evals: &[EF],
) -> Vec<SparseStatement<EF>> {
    let natural_point =
        natural_ordering_point_for_session(sumcheck_air_point, table.padded_rows().trailing_zeros() as usize);
    whir_statements_from_public(
        table.shape,
        table.padded_rows(),
        global_n_vars,
        &natural_point,
        col_evals,
        table.commitments.commitment_root,
    )
}

fn whir_statements_from_public(
    shape: LeanAirShape,
    poseidon_rows: usize,
    global_n_vars: usize,
    natural_point: &[EF],
    col_evals: &[EF],
    commitment_root: [F; DIGEST_LEN],
) -> Vec<SparseStatement<EF>> {
    let mut statements = Vec::with_capacity(2 + DIGEST_LEN);
    statements.push(SparseStatement::new(
        global_n_vars,
        MultilinearPoint(natural_point.to_vec()),
        col_evals
            .iter()
            .copied()
            .enumerate()
            .map(|(selector, value)| SparseValue::new(selector, value))
            .collect(),
    ));

    let final_row = active_poseidon_rows_for_shape(shape) - 1;
    for (limb, &value) in commitment_root.iter().enumerate() {
        let column = DEDICATED_HASH_OUTPUT_START + limb;
        statements.push(SparseStatement::unique_value(
            global_n_vars,
            column * poseidon_rows + final_row,
            EF::from(value),
        ));
    }
    statements
}

fn public_scalars(table: &LeanAirDedicatedHashTable) -> Vec<F> {
    public_scalars_from_shape(
        table.shape,
        table.commitments.commitment_root,
        table.padded_rows(),
        table.committed_columns().len(),
        log2_ceil(table.committed_columns().len() * table.padded_rows()),
    )
}

fn public_scalars_from_shape(
    shape: LeanAirShape,
    commitment_root: [F; DIGEST_LEN],
    poseidon_rows: usize,
    poseidon_columns: usize,
    global_n_vars: usize,
) -> Vec<F> {
    let mut out = Vec::with_capacity(5 + DIGEST_LEN);
    out.push(F::from_usize(shape.log_m));
    out.push(F::from_usize(shape.n_rows));
    out.push(F::from_usize(shape.cell_len_ext));
    out.push(F::from_usize(poseidon_rows));
    out.push(F::from_usize(poseidon_columns));
    out.push(F::from_usize(global_n_vars));
    out.extend_from_slice(&commitment_root);
    out
}

// The row RLC challenge is sampled from the public data commitment root, which is
// itself constrained as the final hash output of the dedicated Poseidon trace.
fn row_coefficient_alpha(shape: LeanAirShape, commitment_root: [F; DIGEST_LEN]) -> EF {
    let mut input = [F::ZERO; 2 * DIGEST_LEN];
    input[0] = F::from_usize(0x1EAD_A1);
    input[1] = F::from_usize(shape.log_m);
    input[2] = F::from_usize(shape.n_rows);
    input[3] = F::from_usize(shape.cell_len_ext);
    input[4..4 + DIGEST_LEN].copy_from_slice(&commitment_root);

    let digest = utils::poseidon16_compress(input);
    let alpha = EF::from_basis_coefficients_fn(|limb| digest[limb]);
    if alpha == EF::ZERO { EF::ONE } else { alpha }
}

fn extra_data_for_air(air_alpha: EF) -> ExtraDataForBuses<EF> {
    ExtraDataForBuses::new(
        vec![EF::ZERO; lean_vm::max_bus_width_including_domainsep().next_power_of_two()],
        EF::ZERO,
        powers(air_alpha, LeanAirDedicatedHashAir.n_constraints() + 1),
    )
}

const DEDICATED_AIR_ENDIANNESS_PIVOT: usize = 12;
const DEDICATED_HASH_AIR_DEGREE: usize = 9;
const DEDICATED_HASH_LOW_DEGREE: usize = 3;
const DEDICATED_HASH_LOW_FULL_EVALS: usize = DEDICATED_HASH_LOW_DEGREE + 1;

#[derive(Debug)]
struct DedicatedHashAirSumcheckSession<'a> {
    multilinears: MleGroup<'a, EF>,
    eq_factor: Vec<EF>,
    current_unpadded_len: usize,
    sum: EF,
    missing_mul_factor: EF,
    extra_data: ExtraDataForBuses<EF>,
    initial_n_vars: usize,
    constraints_eval_at_padding: EF,
    rounds_done: usize,
}

impl<'a> DedicatedHashAirSumcheckSession<'a> {
    fn new(
        packed_multilinears: MleGroup<'a, EF>,
        eq_factor: Vec<EF>,
        extra_data: ExtraDataForBuses<EF>,
        non_padded_n_rows: usize,
    ) -> Self {
        let initial_n_vars = packed_multilinears.n_vars();
        assert_eq!(eq_factor.len(), initial_n_vars);

        let last_point = dedicated_air_column_evals(&packed_multilinears.by_ref(), (1 << initial_n_vars) - 1);
        let constraints_eval_at_padding = LeanAirDedicatedHashAir.eval_extension(&last_point, &extra_data);

        let pivot = DEDICATED_AIR_ENDIANNESS_PIVOT.min(initial_n_vars);
        let has_packed_phase = pivot > packing_log_width::<EF>();
        let padded_n_rows = non_padded_n_rows
            .next_multiple_of(1usize << pivot)
            .min(1usize << initial_n_vars);

        let multilinears = match (packed_multilinears.by_ref(), has_packed_phase) {
            (MleGroupRef::BasePacked(cols), true) => {
                let _span = info_span!("lean-air dedicated chunk-bit-reversing columns").entered();
                let chunk_size = 1usize << pivot;
                let shift = usize::BITS as usize - pivot;
                let bit_reversed = cols
                    .par_iter()
                    .map(|&src| {
                        let mut dst: Vec<PFPacking<EF>> = unsafe { uninitialized_vec(src.len()) };
                        let src_u = PFPacking::<EF>::unpack_slice(src);
                        let dst_u = PFPacking::<EF>::unpack_slice_mut(&mut dst);
                        for (src_chunk, dst_chunk) in
                            src_u.chunks_exact(chunk_size).zip(dst_u.chunks_exact_mut(chunk_size))
                        {
                            for (p, slot) in dst_chunk.iter_mut().enumerate() {
                                *slot = src_chunk[p.reverse_bits() >> shift];
                            }
                        }
                        dst
                    })
                    .collect();
                MleGroup::Owned(MleGroupOwned::BasePacked(bit_reversed))
            }
            _ => unreachable!(),
        };

        Self {
            multilinears,
            eq_factor,
            current_unpadded_len: padded_n_rows,
            sum: EF::ZERO,
            missing_mul_factor: EF::ONE,
            extra_data,
            initial_n_vars,
            constraints_eval_at_padding,
            rounds_done: 0,
        }
    }

    fn pivot(&self) -> usize {
        DEDICATED_AIR_ENDIANNESS_PIVOT.min(self.initial_n_vars)
    }

    fn folding_bit(&self) -> usize {
        let pivot = self.pivot();
        if self.rounds_done < pivot {
            pivot - 1 - self.rounds_done
        } else {
            0
        }
    }

    fn folding_bit_packed(&self) -> usize {
        let bit = self.folding_bit();
        if self.in_phase_1() {
            bit - packing_log_width::<EF>()
        } else {
            bit
        }
    }

    fn in_phase_1(&self) -> bool {
        let w = packing_log_width::<EF>();
        self.rounds_done + w < self.pivot() && self.rounds_done + w + 1 < self.initial_n_vars
    }

    fn active_count_pairs(&self) -> usize {
        if self.in_phase_1() {
            (self.current_unpadded_len / 2) >> packing_log_width::<EF>()
        } else {
            self.current_unpadded_len.div_ceil(2)
        }
    }

    fn permuted_alphas(&self, len: usize) -> Vec<EF> {
        let head_len = (self.initial_n_vars - self.pivot()).min(len);
        let base = &self.eq_factor[..len];
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&base[..head_len]);
        out.extend(base[head_len..].iter().rev().copied());
        out
    }

    fn padding_eq_sum(&self, unpadded_len: usize) -> EF {
        let len = self.initial_n_vars - self.rounds_done;
        let mut alphas = self.permuted_alphas(len);
        alphas[len - 1 - self.folding_bit()] = EF::ZERO;
        mle_of_zeros_then_ones(unpadded_len, &alphas)
    }
}

impl OuterSumcheckSession<EF> for DedicatedHashAirSumcheckSession<'_> {
    fn label(&self) -> &'static str {
        "dedicated hash air"
    }

    fn initial_n_vars(&self) -> usize {
        self.initial_n_vars
    }

    fn sum(&self) -> EF {
        self.sum
    }

    fn bare_degree(&self) -> usize {
        DEDICATED_HASH_AIR_DEGREE
    }

    fn eq_alpha(&self) -> EF {
        *self.eq_factor.last().unwrap()
    }

    fn compute_bare_round_poly(&mut self) -> DensePolynomial<EF> {
        let split_eq = SplitEq::new(&self.permuted_alphas(self.initial_n_vars - self.rounds_done - 1));
        let active_count_pairs = self.active_count_pairs();
        let storage_shift = if self.in_phase_1() {
            packing_log_width::<EF>()
        } else {
            0
        };
        let iter_count_pairs = 1usize << (self.initial_n_vars - self.rounds_done - 1 - storage_shift);
        debug_assert!(active_count_pairs <= iter_count_pairs);

        let padding_contribution = if active_count_pairs < iter_count_pairs {
            self.constraints_eval_at_padding * self.padding_eq_sum(self.current_unpadded_len)
        } else {
            EF::ZERO
        };

        let p_evals_raw = dedicated_hash_compute_raw_poly(
            &self.multilinears.by_ref(),
            &self.extra_data,
            &split_eq,
            self.folding_bit_packed(),
            active_count_pairs,
        );
        let mut p_evals: Vec<EF> = p_evals_raw
            .into_iter()
            .map(|v| (v + padding_contribution) * self.missing_mul_factor)
            .collect();

        let p_at_1 = (self.sum - (EF::ONE - self.eq_alpha()) * p_evals[0]) / self.eq_alpha();
        p_evals.insert(1, p_at_1);

        DensePolynomial::lagrange_interpolation(
            &p_evals
                .iter()
                .enumerate()
                .map(|(i, &val)| (F::from_usize(i), val))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn process_challenge(&mut self, challenge: EF, bare_poly: &DensePolynomial<EF>) {
        let alpha_fold = self.eq_alpha();
        let eq_eval = (EF::ONE - alpha_fold) * (EF::ONE - challenge) + alpha_fold * challenge;
        self.sum = bare_poly.evaluate(challenge) * eq_eval;
        self.missing_mul_factor *= eq_eval;

        let was_in_phase_1 = self.in_phase_1();
        let fold_bit = self.folding_bit_packed();
        self.multilinears = self.multilinears.by_ref().fold_at_bit(challenge, fold_bit).into();

        self.current_unpadded_len = self.current_unpadded_len.div_ceil(2);
        self.rounds_done += 1;
        self.eq_factor.pop();

        if was_in_phase_1 && !self.in_phase_1() {
            self.multilinears = self.multilinears.by_ref().unpack().as_owned_or_clone().into();
        }
    }

    fn final_column_evals(&self) -> Vec<EF> {
        dedicated_air_column_evals(&self.multilinears.by_ref(), 0)
    }
}

fn dedicated_air_column_evals(multilinears: &MleGroupRef<'_, EF>, i: usize) -> Vec<EF> {
    match multilinears {
        MleGroupRef::Base(cols) => cols.iter().map(|c| EF::from(c[i])).collect(),
        MleGroupRef::Extension(cols) => cols.iter().map(|c| c[i]).collect(),
        MleGroupRef::BasePacked(cols) => {
            let (packed_i, lane) = (i >> packing_log_width::<EF>(), i & (packing_width::<EF>() - 1));
            cols.iter().map(|c| EF::from(c[packed_i].as_slice()[lane])).collect()
        }
        MleGroupRef::ExtensionPacked(cols) => {
            let (packed_i, lane) = (i >> packing_log_width::<EF>(), i & (packing_width::<EF>() - 1));
            cols.iter()
                .map(|c| {
                    <EFPacking<EF> as PackedFieldExtension<F, EF>>::to_ext_iter([c[packed_i]])
                        .nth(lane)
                        .unwrap()
                })
                .collect()
        }
    }
}

fn dedicated_hash_compute_raw_poly(
    multilinears: &MleGroupRef<'_, EF>,
    extra_data: &ExtraDataForBuses<EF>,
    split_eq: &SplitEq<EF>,
    fold_bit: usize,
    active_count_pairs: usize,
) -> Vec<EF> {
    match multilinears {
        MleGroupRef::BasePacked(cols) => dedicated_hash_compute_raw_poly_degree_split::<PFPacking<EF>, _, _>(
            cols,
            |j| split_eq.get_packed(j),
            extra_data,
            fold_bit,
            active_count_pairs,
            |s| <EFPacking<EF> as PackedFieldExtension<F, EF>>::to_ext_iter([s]).sum::<EF>(),
        ),
        MleGroupRef::ExtensionPacked(cols) => dedicated_hash_compute_raw_poly_degree_split::<EFPacking<EF>, _, _>(
            cols,
            |j| split_eq.get_packed(j),
            extra_data,
            fold_bit,
            active_count_pairs,
            |s| <EFPacking<EF> as PackedFieldExtension<F, EF>>::to_ext_iter([s]).sum::<EF>(),
        ),
        MleGroupRef::Base(cols) => dedicated_hash_compute_raw_poly_full::<PF<EF>, EF, _, _, _>(
            cols,
            |j| split_eq.get_unpacked(j),
            extra_data,
            fold_bit,
            active_count_pairs,
            |point, extra_data| LeanAirDedicatedHashAir.eval_base(point, extra_data),
            |s| s,
        ),
        MleGroupRef::Extension(cols) => dedicated_hash_compute_raw_poly_full::<EF, EF, _, _, _>(
            cols,
            |j| split_eq.get_unpacked(j),
            extra_data,
            fold_bit,
            active_count_pairs,
            |point, extra_data| LeanAirDedicatedHashAir.eval_extension(point, extra_data),
            |s| s,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn dedicated_hash_compute_raw_poly_degree_split<IF, GetEq, UnpackSum>(
    cols: &[&[IF]],
    get_split_eq: GetEq,
    extra_data: &ExtraDataForBuses<EF>,
    fold_bit: usize,
    active_count_pairs: usize,
    unpack_sum: UnpackSum,
) -> Vec<EF>
where
    IF: Algebra<PFPacking<EF>> + Copy + Send + Sync + Sub<Output = IF> + AddAssign + PrimeCharacteristicRing + 'static,
    EFPacking<EF>: PrimeCharacteristicRing
        + Mul<IF, Output = EFPacking<EF>>
        + Add<IF, Output = EFPacking<EF>>
        + Mul<PFPacking<EF>, Output = EFPacking<EF>>,
    GetEq: Fn(usize) -> EFPacking<EF> + Sync + Send,
    UnpackSum: Fn(EFPacking<EF>) -> EF + Sync + Send,
{
    debug_assert_eq!(cols.len(), DEDICATED_HASH_NUM_COLS);
    let stride = 1usize << fold_bit;
    let lo_mask = stride - 1;
    let n_full = DEDICATED_HASH_LOW_FULL_EVALS;
    let n_skip = DEDICATED_HASH_AIR_DEGREE - n_full;
    let low_n_constraints = LeanAirDedicatedHashAir.low_degree_air().unwrap().1;

    let low_zs = [F::ZERO, F::from_usize(2), F::from_usize(3), F::from_usize(4)];
    let hi_zs = [
        F::from_usize(5),
        F::from_usize(6),
        F::from_usize(7),
        F::from_usize(8),
        F::from_usize(9),
    ];
    let hi_zs_halved = hi_zs.map(|tz| tz.halve());
    let lagrange_coeffs = lagrange_basis_evals(&low_zs, &hi_zs);

    let acc = (0..active_count_pairs)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![EFPacking::<EF>::ZERO; DEDICATED_HASH_AIR_DEGREE],
                    vec![IF::ZERO; DEDICATED_HASH_NUM_COLS],
                    vec![IF::ZERO; DEDICATED_HASH_NUM_COLS],
                    vec![EFPacking::<EF>::ZERO; DEDICATED_HASH_LOW_FULL_EVALS],
                    Vec::<IF>::with_capacity(DEDICATED_HASH_WIDTH),
                    Vec::<IF>::with_capacity(DEDICATED_HASH_WIDTH),
                    Vec::<IF>::with_capacity(DEDICATED_HASH_WIDTH),
                )
            },
            |(mut acc, mut point, mut diff, mut low_evals, mut state_0, mut state_2, mut cached_buf), new_j| {
                let i_hi = new_j >> fold_bit;
                let i_lo = new_j & lo_mask;
                let i0 = (i_hi << (fold_bit + 1)) | i_lo;
                let i1 = i0 | stride;
                let partial_eq = get_split_eq(new_j);

                for (k, c) in cols.iter().enumerate() {
                    let lo = c[i0];
                    let hi = c[i1];
                    point[k] = lo;
                    diff[k] = hi - lo;
                }

                {
                    let mut folder = ConstraintFolderPacked::new(&point[..], &[], extra_data);
                    folder.cached_state = Some(std::mem::take(&mut state_0));
                    Air::eval(&LeanAirDedicatedHashAir, &mut folder, extra_data);
                    acc[0] += folder.accumulator * partial_eq;
                    low_evals[0] = folder.accumulator_low;
                    state_0 = folder.cached_state.unwrap();
                }

                for k in 0..DEDICATED_HASH_NUM_COLS {
                    point[k] += diff[k].double();
                }
                {
                    let mut folder = ConstraintFolderPacked::new(&point[..], &[], extra_data);
                    folder.cached_state = Some(std::mem::take(&mut state_2));
                    Air::eval(&LeanAirDedicatedHashAir, &mut folder, extra_data);
                    acc[1] += folder.accumulator * partial_eq;
                    low_evals[1] = folder.accumulator_low;
                    state_2 = folder.cached_state.unwrap();
                }

                for z_idx in 2..n_full {
                    for k in 0..DEDICATED_HASH_NUM_COLS {
                        point[k] += diff[k];
                    }
                    let mut folder = ConstraintFolderPacked::new(&point[..], &[], extra_data);
                    Air::eval(&LeanAirDedicatedHashAir, &mut folder, extra_data);
                    acc[z_idx] += folder.accumulator * partial_eq;
                    low_evals[z_idx] = folder.accumulator_low;
                }

                for t in 0..n_skip {
                    for k in 0..DEDICATED_HASH_NUM_COLS {
                        point[k] += diff[k];
                    }

                    cached_buf.resize(DEDICATED_HASH_WIDTH, IF::ZERO);
                    let z = PFPacking::<EF>::from(hi_zs_halved[t]);
                    for i in 0..DEDICATED_HASH_WIDTH {
                        cached_buf[i] = state_0[i] + (state_2[i] - state_0[i]) * z;
                    }

                    let mut folder = ConstraintFolderPacked::new(&point[..], &[], extra_data);
                    folder.skip_low = true;
                    folder.cached_state = Some(std::mem::take(&mut cached_buf));
                    folder.low_ci_count = low_n_constraints;
                    Air::eval(&LeanAirDedicatedHashAir, &mut folder, extra_data);
                    cached_buf = folder.cached_state.unwrap();

                    let mut low_interpolated = EFPacking::<EF>::ZERO;
                    for (i, lc) in lagrange_coeffs[t].iter().enumerate() {
                        low_interpolated += low_evals[i] * PFPacking::<EF>::from(*lc);
                    }

                    acc[n_full + t] += (folder.accumulator + low_interpolated) * partial_eq;
                }

                (acc, point, diff, low_evals, state_0, state_2, cached_buf)
            },
        )
        .map(|(acc, ..)| acc)
        .reduce(
            || vec![EFPacking::<EF>::ZERO; DEDICATED_HASH_AIR_DEGREE],
            |mut a, b| {
                for i in 0..DEDICATED_HASH_AIR_DEGREE {
                    a[i] += b[i];
                }
                a
            },
        );

    acc.into_iter().map(unpack_sum).collect()
}

#[allow(clippy::too_many_arguments)]
fn dedicated_hash_compute_raw_poly_full<IF, EFT, GetEq, EvalFn, UnpackSum>(
    cols: &[&[IF]],
    get_split_eq: GetEq,
    extra_data: &ExtraDataForBuses<EF>,
    fold_bit: usize,
    active_count_pairs: usize,
    eval_fn: EvalFn,
    unpack_sum: UnpackSum,
) -> Vec<EF>
where
    IF: Copy + Send + Sync + Sub<Output = IF> + AddAssign + PrimeCharacteristicRing,
    EFT: Copy + Send + Sync + Add<Output = EFT> + AddAssign + Mul<Output = EFT> + PrimeCharacteristicRing,
    GetEq: Fn(usize) -> EFT + Sync + Send,
    EvalFn: Fn(&[IF], &ExtraDataForBuses<EF>) -> EFT + Sync + Send,
    UnpackSum: Fn(EFT) -> EF + Sync + Send,
{
    debug_assert_eq!(cols.len(), DEDICATED_HASH_NUM_COLS);
    let stride = 1usize << fold_bit;
    let lo_mask = stride - 1;

    let acc = (0..active_count_pairs)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![EFT::ZERO; DEDICATED_HASH_AIR_DEGREE],
                    vec![IF::ZERO; DEDICATED_HASH_NUM_COLS],
                    vec![IF::ZERO; DEDICATED_HASH_NUM_COLS],
                )
            },
            |(mut acc, mut point, mut diff), new_j| {
                let i_hi = new_j >> fold_bit;
                let i_lo = new_j & lo_mask;
                let i0 = (i_hi << (fold_bit + 1)) | i_lo;
                let i1 = i0 | stride;
                let partial_eq = get_split_eq(new_j);

                for (k, c) in cols.iter().enumerate() {
                    let lo = c[i0];
                    let hi = c[i1];
                    point[k] = lo;
                    diff[k] = hi - lo;
                }

                acc[0] += eval_fn(&point, extra_data) * partial_eq;
                for k in 0..DEDICATED_HASH_NUM_COLS {
                    point[k] += diff[k];
                }
                for acc_z in &mut acc[1..] {
                    for k in 0..DEDICATED_HASH_NUM_COLS {
                        point[k] += diff[k];
                    }
                    *acc_z += eval_fn(&point, extra_data) * partial_eq;
                }

                (acc, point, diff)
            },
        )
        .map(|(acc, _, _)| acc)
        .reduce(
            || vec![EFT::ZERO; DEDICATED_HASH_AIR_DEGREE],
            |mut a, b| {
                for i in 0..DEDICATED_HASH_AIR_DEGREE {
                    a[i] += b[i];
                }
                a
            },
        );

    acc.into_iter().map(unpack_sum).collect()
}

const VIRTUAL_LINEAR_BUS_BARE_DEGREE: usize = 2;

#[derive(Clone, Copy, Debug)]
struct SparseCoeff {
    row: usize,
    coeff: EF,
}

#[derive(Debug)]
struct VirtualLinearBusSession {
    values: Vec<Vec<EF>>,
    coefficients: Vec<Vec<SparseCoeff>>,
    current_n_vars: usize,
    initial_n_vars: usize,
    sum: EF,
    missing_mul_factor: EF,
}

impl VirtualLinearBusSession {
    fn new(
        committed_columns: &[Vec<F>],
        column_indexes: Vec<usize>,
        coefficients: Vec<Vec<SparseCoeff>>,
        log_rows: usize,
    ) -> Self {
        assert_eq!(column_indexes.len(), coefficients.len());
        assert!(column_indexes.iter().all(|&column| column < committed_columns.len()));
        assert_eq!(committed_columns[0].len(), 1 << log_rows);
        assert!(coefficients.iter().flatten().all(|term| term.row < (1 << log_rows)));

        let values = column_indexes
            .par_iter()
            .map(|&column| committed_columns[column].iter().copied().map(EF::from).collect())
            .collect();
        Self {
            values,
            coefficients,
            current_n_vars: log_rows,
            initial_n_vars: log_rows,
            sum: EF::ZERO,
            missing_mul_factor: EF::ONE,
        }
    }
}

fn virtual_linear_eq_alpha() -> EF {
    EF::from_usize(2).inverse()
}

impl OuterSumcheckSession<EF> for VirtualLinearBusSession {
    fn label(&self) -> &'static str {
        "sparse linear"
    }

    fn initial_n_vars(&self) -> usize {
        self.initial_n_vars
    }

    fn sum(&self) -> EF {
        self.sum
    }

    fn bare_degree(&self) -> usize {
        VIRTUAL_LINEAR_BUS_BARE_DEGREE
    }

    fn eq_alpha(&self) -> EF {
        virtual_linear_eq_alpha()
    }

    fn compute_bare_round_poly(&mut self) -> DensePolynomial<EF> {
        let current_len = self.values[0].len();
        debug_assert!(current_len > 1);
        debug_assert_eq!(current_len, 1 << self.current_n_vars);
        let partial_eq = constant_half_eq_eval(self.current_n_vars - 1);

        let evals = self
            .values
            .par_iter()
            .zip(&self.coefficients)
            .map(|(values, coefficients)| {
                let mut evals = [EF::ZERO; VIRTUAL_LINEAR_BUS_BARE_DEGREE + 1];
                let mut idx = 0;
                while idx < coefficients.len() {
                    let parent = coefficients[idx].row >> 1;
                    let mut c0 = EF::ZERO;
                    let mut c1 = EF::ZERO;
                    while idx < coefficients.len() && (coefficients[idx].row >> 1) == parent {
                        if coefficients[idx].row & 1 == 0 {
                            c0 += coefficients[idx].coeff;
                        } else {
                            c1 += coefficients[idx].coeff;
                        }
                        idx += 1;
                    }

                    let v0 = values[2 * parent];
                    let v_delta = values[2 * parent + 1] - v0;
                    let c_delta = c1 - c0;

                    evals[0] += partial_eq * v0 * c0;
                    evals[1] += partial_eq * (v0 + v_delta) * (c0 + c_delta);
                    evals[2] += partial_eq * (v0 + v_delta.double()) * (c0 + c_delta.double());
                }
                evals
            })
            .reduce(
                || [EF::ZERO; VIRTUAL_LINEAR_BUS_BARE_DEGREE + 1],
                |mut a, b| {
                    for (lhs, rhs) in a.iter_mut().zip(b) {
                        *lhs += rhs;
                    }
                    a
                },
            );

        let evals = evals.map(|eval| eval * self.missing_mul_factor);
        DensePolynomial::lagrange_interpolation(&[
            (F::ZERO, evals[0]),
            (F::ONE, evals[1]),
            (F::from_usize(2), evals[2]),
        ])
        .unwrap()
    }

    fn process_challenge(&mut self, challenge: EF, bare_poly: &DensePolynomial<EF>) {
        let eq_alpha = self.eq_alpha();
        let eq_eval = (EF::ONE - eq_alpha) * (EF::ONE - challenge) + eq_alpha * challenge;
        self.sum = bare_poly.evaluate(challenge) * eq_eval;
        self.missing_mul_factor *= eq_eval;

        fold_adjacent(&mut self.values, challenge);
        fold_sparse_coefficients(&mut self.coefficients, challenge);
        self.current_n_vars -= 1;
    }

    fn final_column_evals(&self) -> Vec<EF> {
        self.values.iter().map(|column| column[0]).collect()
    }
}

#[derive(Debug)]
struct DenseLinearBusSession {
    values: Vec<Vec<EF>>,
    coefficients: Vec<Vec<EF>>,
    current_n_vars: usize,
    initial_n_vars: usize,
    sum: EF,
    missing_mul_factor: EF,
}

impl DenseLinearBusSession {
    fn new(
        committed_columns: &[Vec<F>],
        column_indexes: Vec<usize>,
        coefficients: Vec<Vec<EF>>,
        log_rows: usize,
    ) -> Self {
        assert_eq!(column_indexes.len(), coefficients.len());
        assert!(column_indexes.iter().all(|&column| column < committed_columns.len()));
        assert_eq!(committed_columns[0].len(), 1 << log_rows);
        assert!(coefficients.iter().all(|column| column.len() == (1 << log_rows)));

        let values = column_indexes
            .par_iter()
            .map(|&column| committed_columns[column].iter().copied().map(EF::from).collect())
            .collect();
        Self {
            values,
            coefficients,
            current_n_vars: log_rows,
            initial_n_vars: log_rows,
            sum: EF::ZERO,
            missing_mul_factor: EF::ONE,
        }
    }
}

impl OuterSumcheckSession<EF> for DenseLinearBusSession {
    fn label(&self) -> &'static str {
        "dense cell-chain"
    }

    fn initial_n_vars(&self) -> usize {
        self.initial_n_vars
    }

    fn sum(&self) -> EF {
        self.sum
    }

    fn bare_degree(&self) -> usize {
        VIRTUAL_LINEAR_BUS_BARE_DEGREE
    }

    fn eq_alpha(&self) -> EF {
        virtual_linear_eq_alpha()
    }

    fn compute_bare_round_poly(&mut self) -> DensePolynomial<EF> {
        let current_len = self.values[0].len();
        debug_assert!(current_len > 1);
        debug_assert_eq!(current_len, 1 << self.current_n_vars);
        let partial_eq = constant_half_eq_eval(self.current_n_vars - 1);

        let evals = self
            .values
            .par_iter()
            .zip(&self.coefficients)
            .map(|(values, coefficients)| {
                let mut evals = [EF::ZERO; VIRTUAL_LINEAR_BUS_BARE_DEGREE + 1];
                for parent in 0..(current_len / 2) {
                    let v0 = values[2 * parent];
                    let v_delta = values[2 * parent + 1] - v0;
                    let c0 = coefficients[2 * parent];
                    let c_delta = coefficients[2 * parent + 1] - c0;

                    evals[0] += partial_eq * v0 * c0;
                    evals[1] += partial_eq * (v0 + v_delta) * (c0 + c_delta);
                    evals[2] += partial_eq * (v0 + v_delta.double()) * (c0 + c_delta.double());
                }
                evals
            })
            .reduce(
                || [EF::ZERO; VIRTUAL_LINEAR_BUS_BARE_DEGREE + 1],
                |mut a, b| {
                    for (lhs, rhs) in a.iter_mut().zip(b) {
                        *lhs += rhs;
                    }
                    a
                },
            );

        let evals = evals.map(|eval| eval * self.missing_mul_factor);
        DensePolynomial::lagrange_interpolation(&[
            (F::ZERO, evals[0]),
            (F::ONE, evals[1]),
            (F::from_usize(2), evals[2]),
        ])
        .unwrap()
    }

    fn process_challenge(&mut self, challenge: EF, bare_poly: &DensePolynomial<EF>) {
        let eq_alpha = self.eq_alpha();
        let eq_eval = (EF::ONE - eq_alpha) * (EF::ONE - challenge) + eq_alpha * challenge;
        self.sum = bare_poly.evaluate(challenge) * eq_eval;
        self.missing_mul_factor *= eq_eval;

        fold_adjacent(&mut self.values, challenge);
        fold_adjacent(&mut self.coefficients, challenge);
        self.current_n_vars -= 1;
    }

    fn final_column_evals(&self) -> Vec<EF> {
        self.values.iter().map(|column| column[0]).collect()
    }
}

fn fold_adjacent(columns: &mut [Vec<EF>], challenge: EF) {
    columns.par_iter_mut().for_each(|column| {
        let half_len = column.len() / 2;
        for j in 0..half_len {
            let lo = column[2 * j];
            let hi = column[2 * j + 1];
            column[j] = lo + challenge * (hi - lo);
        }
        column.truncate(half_len);
    });
}

fn fold_sparse_coefficients(columns: &mut [Vec<SparseCoeff>], challenge: EF) {
    columns.par_iter_mut().for_each(|column| {
        let mut folded = Vec::with_capacity(column.len().div_ceil(2));
        let mut idx = 0;
        while idx < column.len() {
            let parent = column[idx].row >> 1;
            let mut c0 = EF::ZERO;
            let mut c1 = EF::ZERO;
            while idx < column.len() && (column[idx].row >> 1) == parent {
                if column[idx].row & 1 == 0 {
                    c0 += column[idx].coeff;
                } else {
                    c1 += column[idx].coeff;
                }
                idx += 1;
            }

            let coeff = c0 + challenge * (c1 - c0);
            if coeff != EF::ZERO {
                folded.push(SparseCoeff { row: parent, coeff });
            }
        }
        *column = folded;
    });
}

fn parity_sparse_coefficients(
    shape: LeanAirShape,
    poseidon_rows: usize,
    row_coeff_alpha: EF,
    parity_point: EF,
) -> Vec<Vec<SparseCoeff>> {
    assert!(poseidon_rows >= active_poseidon_rows_for_shape(shape));

    let (slice_l, slice_r) = barycentric_slices(shape.log_m, parity_point);
    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    let basis = extension_basis();
    let ranges = ScheduleRanges::new(shape);
    let row_weights = powers(row_coeff_alpha, shape.n_rows);

    let coefficients = (0..DEDICATED_HASH_WIDTH)
        .into_par_iter()
        .map(|input_offset| {
            let chunks_for_slot = if input_offset < DIGEST_LEN {
                1
            } else {
                ranges.chunks_per_cell
            };
            let mut column = Vec::with_capacity(shape.n_rows * shape.num_cells() * chunks_for_slot);
            for (row_idx, &row_weight) in row_weights.iter().enumerate() {
                for cell_idx in 0..shape.num_cells() {
                    for chunk_idx in 0..ranges.chunks_per_cell {
                        let Some(base_limb_in_cell) = base_limb_for_input_slot(input_offset, chunk_idx) else {
                            continue;
                        };
                        if base_limb_in_cell >= shape.cell_len_base() {
                            continue;
                        }
                        let term_in_cell = base_limb_in_cell / dim;
                        let limb_idx = base_limb_in_cell % dim;
                        let term_idx = cell_idx * shape.cell_len_ext + term_in_cell;
                        let evaluation_idx = term_idx / 2;
                        let coeff = if term_idx.is_multiple_of(2) {
                            row_weight * slice_l[evaluation_idx]
                        } else {
                            -row_weight * slice_r[evaluation_idx]
                        } * basis[limb_idx];
                        if coeff != EF::ZERO {
                            column.push(SparseCoeff {
                                row: ranges.cell_hash_row(shape, row_idx, cell_idx, chunk_idx),
                                coeff,
                            });
                        }
                    }
                }
            }
            column
        })
        .collect::<Vec<_>>();

    debug_assert!(coefficients.iter().all(|column| is_strictly_sorted_sparse(column)));
    coefficients
}

fn base_limb_for_input_slot(input_offset: usize, chunk_idx: usize) -> Option<usize> {
    if chunk_idx == 0 {
        Some(input_offset)
    } else if input_offset >= DIGEST_LEN {
        Some(2 * DIGEST_LEN + (chunk_idx - 1) * DIGEST_LEN + input_offset - DIGEST_LEN)
    } else {
        None
    }
}

fn mixed_sparse_linear_coefficients(
    parity_coefficients: Vec<Vec<SparseCoeff>>,
    mut link_coefficients: Vec<Vec<SparseCoeff>>,
    link_mix: EF,
) -> Vec<Vec<SparseCoeff>> {
    assert_eq!(parity_coefficients.len(), DEDICATED_HASH_WIDTH);
    assert_eq!(link_coefficients.len(), DEDICATED_HASH_WIDTH + DIGEST_LEN);

    link_coefficients.par_iter_mut().for_each(|column| {
        for term in column {
            term.coeff *= link_mix;
        }
    });
    link_coefficients
        .par_iter_mut()
        .take(DEDICATED_HASH_WIDTH)
        .zip(parity_coefficients.into_par_iter())
        .for_each(|(target, parity)| merge_sorted_sparse_columns(target, parity));
    link_coefficients
}

fn merge_sorted_sparse_columns(target: &mut Vec<SparseCoeff>, source: Vec<SparseCoeff>) {
    let mut merged = Vec::with_capacity(target.len() + source.len());
    let mut lhs = target.iter().copied().peekable();
    let mut rhs = source.into_iter().peekable();

    while let (Some(&left), Some(&right)) = (lhs.peek(), rhs.peek()) {
        match left.row.cmp(&right.row) {
            std::cmp::Ordering::Less => {
                merged.push(left);
                lhs.next();
            }
            std::cmp::Ordering::Greater => {
                merged.push(right);
                rhs.next();
            }
            std::cmp::Ordering::Equal => {
                let coeff = left.coeff + right.coeff;
                if coeff != EF::ZERO {
                    merged.push(SparseCoeff { row: left.row, coeff });
                }
                lhs.next();
                rhs.next();
            }
        }
    }
    merged.extend(lhs);
    merged.extend(rhs);
    *target = merged;
}

fn sparse_linear_sum(
    table: &LeanAirDedicatedHashTable,
    column_indexes: &[usize],
    coefficients: &[Vec<SparseCoeff>],
) -> EF {
    column_indexes
        .iter()
        .zip(coefficients)
        .map(|(&column, coeffs)| {
            coeffs
                .iter()
                .map(|term| term.coeff * table.committed_columns()[column][term.row])
                .sum::<EF>()
        })
        .sum()
}

fn dense_linear_sum(table: &LeanAirDedicatedHashTable, column_indexes: &[usize], coefficients: &[Vec<EF>]) -> EF {
    column_indexes
        .iter()
        .zip(coefficients)
        .map(|(&column, coeffs)| {
            coeffs
                .iter()
                .enumerate()
                .map(|(row, &coeff)| coeff * table.committed_columns()[column][row])
                .sum::<EF>()
        })
        .sum()
}

fn linear_coefficient_evals_from_schedule(
    shape: LeanAirShape,
    poseidon_rows: usize,
    row_coeff_alpha: EF,
    parity_point: EF,
    link_point: EF,
    link_mix: EF,
    sumcheck_point: &[EF],
) -> Vec<EF> {
    assert!(poseidon_rows >= active_poseidon_rows_for_shape(shape));

    let mut point_rev = sumcheck_point.to_vec();
    point_rev.reverse();
    let poseidon_eq = eval_eq(&point_rev);
    assert_eq!(poseidon_eq.len(), poseidon_rows);

    let mut evals = vec![EF::ZERO; DEDICATED_HASH_WIDTH + DIGEST_LEN];
    let (slice_l, slice_r) = barycentric_slices(shape.log_m, parity_point);
    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    let basis = extension_basis();
    let mut row_weight = EF::ONE;

    for row_idx in 0..shape.n_rows {
        for j in 0..shape.message_len_ext() {
            add_codeword_term_coefficient_eval(
                &mut evals,
                shape,
                row_idx,
                2 * j,
                row_weight * slice_l[j],
                &basis,
                dim,
                &poseidon_eq,
            );
            add_codeword_term_coefficient_eval(
                &mut evals,
                shape,
                row_idx,
                2 * j + 1,
                -row_weight * slice_r[j],
                &basis,
                dim,
                &poseidon_eq,
            );
        }
        row_weight *= row_coeff_alpha;
    }

    add_link_coefficient_evals(&mut evals, shape, link_point, link_mix, &poseidon_eq);
    evals
}

fn cell_link_coefficient_evals_from_schedule(
    shape: LeanAirShape,
    poseidon_rows: usize,
    cell_link_point: EF,
    sumcheck_point: &[EF],
) -> Vec<EF> {
    assert!(poseidon_rows >= active_poseidon_rows_for_shape(shape));

    let mut point_rev = sumcheck_point.to_vec();
    point_rev.reverse();
    let poseidon_eq = eval_eq(&point_rev);
    assert_eq!(poseidon_eq.len(), poseidon_rows);

    let mut evals = vec![EF::ZERO; 2 * DIGEST_LEN];
    add_cell_link_coefficient_evals(&mut evals, shape, cell_link_point, &poseidon_eq);
    evals
}

fn add_codeword_term_coefficient_eval(
    evals: &mut [EF],
    shape: LeanAirShape,
    row_idx: usize,
    term_idx: usize,
    coeff: EF,
    basis: &[EF],
    dim: usize,
    poseidon_eq: &[EF],
) {
    for (limb_idx, &basis_limb) in basis.iter().enumerate().take(dim) {
        let (trace_row, input_offset) = codeword_limb_position(shape, row_idx, term_idx, limb_idx);
        evals[input_offset] += coeff * basis_limb * poseidon_eq[trace_row];
    }
}

fn add_link_coefficient_evals(evals: &mut [EF], shape: LeanAirShape, link_point: EF, link_mix: EF, poseidon_eq: &[EF]) {
    let mut next_weight = EF::ONE;
    let ranges = ScheduleRanges::new(shape);

    for row_idx in 0..shape.n_rows {
        for cell_idx in 0..shape.num_systematic_cells() {
            let digest_row = ranges.systematic_row_digest_row(shape, row_idx, cell_idx);
            if cell_idx == 0 {
                add_zero_digest_eval(
                    evals,
                    digest_row,
                    0,
                    &mut next_weight,
                    link_point,
                    link_mix,
                    poseidon_eq,
                );
            } else {
                let previous_row = ranges.systematic_row_digest_row(shape, row_idx, cell_idx - 1);
                add_digest_link_eval(
                    evals,
                    previous_row,
                    digest_row,
                    0,
                    &mut next_weight,
                    link_point,
                    link_mix,
                    poseidon_eq,
                );
            }
            let cell_row = ranges.cell_digest_row(shape, row_idx, cell_idx);
            add_digest_link_eval(
                evals,
                cell_row,
                digest_row,
                DIGEST_LEN,
                &mut next_weight,
                link_point,
                link_mix,
                poseidon_eq,
            );
        }
    }

    for row_idx in 0..shape.n_rows {
        let root_row = ranges.row_root_row(row_idx);
        if row_idx == 0 {
            add_zero_digest_eval(evals, root_row, 0, &mut next_weight, link_point, link_mix, poseidon_eq);
        } else {
            add_digest_link_eval(
                evals,
                ranges.row_root_row(row_idx - 1),
                root_row,
                0,
                &mut next_weight,
                link_point,
                link_mix,
                poseidon_eq,
            );
        }
        let row_digest = ranges.systematic_row_digest_row(shape, row_idx, shape.num_systematic_cells() - 1);
        add_digest_link_eval(
            evals,
            row_digest,
            root_row,
            DIGEST_LEN,
            &mut next_weight,
            link_point,
            link_mix,
            poseidon_eq,
        );
    }

    let row_tree_layers = log2_strict(shape.padded_rows());
    for cell_idx in 0..shape.num_cells() {
        for layer_idx in 0..row_tree_layers {
            let layer_width = shape.padded_rows() >> (layer_idx + 1);
            for node_idx in 0..layer_width {
                let parent_row = ranges.column_merkle_row(shape, cell_idx, layer_idx, node_idx);
                for side in 0..2 {
                    let child_idx = 2 * node_idx + side;
                    let input_offset = side * DIGEST_LEN;
                    if layer_idx == 0 {
                        if child_idx < shape.n_rows {
                            let cell_row = ranges.cell_digest_row(shape, child_idx, cell_idx);
                            add_digest_link_eval(
                                evals,
                                cell_row,
                                parent_row,
                                input_offset,
                                &mut next_weight,
                                link_point,
                                link_mix,
                                poseidon_eq,
                            );
                        } else {
                            add_zero_digest_eval(
                                evals,
                                parent_row,
                                input_offset,
                                &mut next_weight,
                                link_point,
                                link_mix,
                                poseidon_eq,
                            );
                        }
                    } else {
                        let child_row = ranges.column_merkle_row(shape, cell_idx, layer_idx - 1, child_idx);
                        add_digest_link_eval(
                            evals,
                            child_row,
                            parent_row,
                            input_offset,
                            &mut next_weight,
                            link_point,
                            link_mix,
                            poseidon_eq,
                        );
                    }
                }
            }
        }
    }

    let column_tree_layers = log2_strict(shape.num_cells());
    for layer_idx in 0..column_tree_layers {
        let layer_width = shape.num_cells() >> (layer_idx + 1);
        for node_idx in 0..layer_width {
            let parent_row = ranges.column_root_row(shape, layer_idx, node_idx);
            for side in 0..2 {
                let child_idx = 2 * node_idx + side;
                let input_offset = side * DIGEST_LEN;
                let child_row = if layer_idx == 0 {
                    column_root_source_row(shape, ranges, child_idx)
                } else {
                    ranges.column_root_row(shape, layer_idx - 1, child_idx)
                };
                add_digest_link_eval(
                    evals,
                    child_row,
                    parent_row,
                    input_offset,
                    &mut next_weight,
                    link_point,
                    link_mix,
                    poseidon_eq,
                );
            }
        }
    }

    add_digest_link_eval(
        evals,
        ranges.row_root_row(shape.n_rows - 1),
        ranges.final_root_row,
        0,
        &mut next_weight,
        link_point,
        link_mix,
        poseidon_eq,
    );
    add_digest_link_eval(
        evals,
        column_commitment_source_row(shape, ranges),
        ranges.final_root_row,
        DIGEST_LEN,
        &mut next_weight,
        link_point,
        link_mix,
        poseidon_eq,
    );
}

fn add_cell_link_coefficient_evals(evals: &mut [EF], shape: LeanAirShape, cell_link_point: EF, poseidon_eq: &[EF]) {
    let mut next_weight = EF::ONE;
    let ranges = ScheduleRanges::new(shape);

    for row_idx in 0..shape.n_rows {
        for cell_idx in 0..shape.num_cells() {
            for chunk_idx in 1..ranges.chunks_per_cell {
                let previous_row = ranges.cell_hash_row(shape, row_idx, cell_idx, chunk_idx - 1);
                let current_row = ranges.cell_hash_row(shape, row_idx, cell_idx, chunk_idx);
                for limb in 0..DIGEST_LEN {
                    let weight = take_link_weight(&mut next_weight, cell_link_point);
                    evals[limb] += weight * poseidon_eq[previous_row];
                    evals[DIGEST_LEN + limb] -= weight * poseidon_eq[current_row];
                }
            }
        }
    }
}

fn add_digest_link_eval(
    evals: &mut [EF],
    source_row: usize,
    sink_row: usize,
    sink_offset: usize,
    next_weight: &mut EF,
    link_point: EF,
    link_mix: EF,
    poseidon_eq: &[EF],
) {
    for limb in 0..DIGEST_LEN {
        let weight = link_mix * take_link_weight(next_weight, link_point);
        add_linear_link_eval(
            evals,
            TracePosition {
                row: source_row,
                column: DEDICATED_HASH_OUTPUT_START + limb,
            },
            TracePosition {
                row: sink_row,
                column: DEDICATED_HASH_INPUT_START + sink_offset + limb,
            },
            weight,
            poseidon_eq,
        );
    }
}

fn add_zero_digest_eval(
    evals: &mut [EF],
    sink_row: usize,
    sink_offset: usize,
    next_weight: &mut EF,
    link_point: EF,
    link_mix: EF,
    poseidon_eq: &[EF],
) {
    for limb in 0..DIGEST_LEN {
        let weight = link_mix * take_link_weight(next_weight, link_point);
        add_zero_link_eval(
            evals,
            TracePosition {
                row: sink_row,
                column: DEDICATED_HASH_INPUT_START + sink_offset + limb,
            },
            weight,
            poseidon_eq,
        );
    }
}

fn add_linear_link_eval(evals: &mut [EF], source: TracePosition, sink: TracePosition, weight: EF, poseidon_eq: &[EF]) {
    evals[link_column_slot(source.column)] += weight * poseidon_eq[source.row];
    evals[link_column_slot(sink.column)] -= weight * poseidon_eq[sink.row];
}

fn add_zero_link_eval(evals: &mut [EF], position: TracePosition, weight: EF, poseidon_eq: &[EF]) {
    evals[link_column_slot(position.column)] += weight * poseidon_eq[position.row];
}

fn add_sparse_term(coefficients: &mut [Vec<SparseCoeff>], slot: usize, row: usize, coeff: EF) {
    if coeff != EF::ZERO {
        coefficients[slot].push(SparseCoeff { row, coeff });
    }
}

fn is_strictly_sorted_sparse(column: &[SparseCoeff]) -> bool {
    column.windows(2).all(|pair| pair[0].row < pair[1].row)
}

fn merge_sparse_coefficients(coefficients: &mut [Vec<SparseCoeff>]) {
    for column in coefficients {
        column.sort_unstable_by_key(|term| term.row);
        let mut merged: Vec<SparseCoeff> = Vec::with_capacity(column.len());
        for term in column.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.row == term.row
            {
                last.coeff += term.coeff;
                continue;
            }
            merged.push(term);
        }
        merged.retain(|term| term.coeff != EF::ZERO);
        *column = merged;
    }
}

#[derive(Clone, Copy, Debug)]
struct TracePosition {
    row: usize,
    column: usize,
}

#[derive(Clone, Copy, Debug)]
struct ScheduleRanges {
    cell_start: usize,
    systematic_row_start: usize,
    row_root_start: usize,
    column_merkle_start: usize,
    column_root_start: usize,
    final_root_row: usize,
    chunks_per_cell: usize,
}

impl ScheduleRanges {
    fn new(shape: LeanAirShape) -> Self {
        let chunks_per_cell = chunks_per_cell(shape);
        let cell_count = shape.n_rows * shape.num_cells() * chunks_per_cell;
        let systematic_count = shape.n_rows * shape.num_systematic_cells();
        let row_root_count = shape.n_rows;
        let column_merkle_count = shape.num_cells() * (shape.padded_rows() - 1);
        let column_root_count = shape.num_cells() - 1;

        let cell_start = 0;
        let systematic_row_start = cell_start + cell_count;
        let row_root_start = systematic_row_start + systematic_count;
        let column_merkle_start = row_root_start + row_root_count;
        let column_root_start = column_merkle_start + column_merkle_count;
        let final_root_row = column_root_start + column_root_count;

        Self {
            cell_start,
            systematic_row_start,
            row_root_start,
            column_merkle_start,
            column_root_start,
            final_root_row,
            chunks_per_cell,
        }
    }

    fn cell_hash_row(self, shape: LeanAirShape, row_idx: usize, cell_idx: usize, chunk_idx: usize) -> usize {
        self.cell_start + (row_idx * shape.num_cells() + cell_idx) * self.chunks_per_cell + chunk_idx
    }

    fn cell_digest_row(self, shape: LeanAirShape, row_idx: usize, cell_idx: usize) -> usize {
        self.cell_hash_row(shape, row_idx, cell_idx, self.chunks_per_cell - 1)
    }

    fn systematic_row_digest_row(self, shape: LeanAirShape, row_idx: usize, cell_idx: usize) -> usize {
        self.systematic_row_start + row_idx * shape.num_systematic_cells() + cell_idx
    }

    fn row_root_row(self, row_idx: usize) -> usize {
        self.row_root_start + row_idx
    }

    fn column_merkle_row(self, shape: LeanAirShape, cell_idx: usize, layer_idx: usize, node_idx: usize) -> usize {
        let padded_rows = shape.padded_rows();
        self.column_merkle_start + cell_idx * (padded_rows - 1) + padded_rows - (padded_rows >> layer_idx) + node_idx
    }

    fn column_root_row(self, shape: LeanAirShape, layer_idx: usize, node_idx: usize) -> usize {
        let num_cells = shape.num_cells();
        self.column_root_start + num_cells - (num_cells >> layer_idx) + node_idx
    }
}

fn link_column_indexes() -> Vec<usize> {
    (0..DEDICATED_HASH_WIDTH)
        .map(|offset| DEDICATED_HASH_INPUT_START + offset)
        .chain((0..DIGEST_LEN).map(|offset| DEDICATED_HASH_OUTPUT_START + offset))
        .collect()
}

fn cell_link_column_indexes() -> Vec<usize> {
    (0..DIGEST_LEN)
        .map(|offset| DEDICATED_HASH_OUTPUT_START + offset)
        .chain((0..DIGEST_LEN).map(|offset| DEDICATED_HASH_INPUT_START + offset))
        .collect()
}

fn cell_link_dense_coefficients(shape: LeanAirShape, poseidon_rows: usize, cell_link_point: EF) -> Vec<Vec<EF>> {
    assert!(poseidon_rows >= active_poseidon_rows_for_shape(shape));

    let mut coefficients = vec![EF::zero_vec(poseidon_rows); 2 * DIGEST_LEN];
    let ranges = ScheduleRanges::new(shape);
    let limb_starts = powers(cell_link_point, DIGEST_LEN);
    let limb_step = cell_link_point.exp_u64(DIGEST_LEN as u64);

    let (output_coefficients, input_coefficients) = coefficients.split_at_mut(DIGEST_LEN);
    output_coefficients
        .par_iter_mut()
        .zip(input_coefficients.par_iter_mut())
        .enumerate()
        .for_each(|(limb, (output_coefficients, input_coefficients))| {
            let mut weight = limb_starts[limb];
            for row_idx in 0..shape.n_rows {
                for cell_idx in 0..shape.num_cells() {
                    for chunk_idx in 1..ranges.chunks_per_cell {
                        let previous_row = ranges.cell_hash_row(shape, row_idx, cell_idx, chunk_idx - 1);
                        let current_row = ranges.cell_hash_row(shape, row_idx, cell_idx, chunk_idx);
                        output_coefficients[previous_row] += weight;
                        input_coefficients[current_row] -= weight;
                        weight *= limb_step;
                    }
                }
            }
        });

    coefficients
}

fn link_sparse_coefficients(shape: LeanAirShape, poseidon_rows: usize, link_point: EF) -> Vec<Vec<SparseCoeff>> {
    assert!(poseidon_rows >= active_poseidon_rows_for_shape(shape));

    let mut coefficients = vec![Vec::new(); DEDICATED_HASH_WIDTH + DIGEST_LEN];
    reserve_link_sparse_coefficients(&mut coefficients, shape);
    let mut next_weight = EF::ONE;
    let ranges = ScheduleRanges::new(shape);

    for row_idx in 0..shape.n_rows {
        for cell_idx in 0..shape.num_systematic_cells() {
            let digest_row = ranges.systematic_row_digest_row(shape, row_idx, cell_idx);
            if cell_idx == 0 {
                add_zero_digest(&mut coefficients, digest_row, 0, &mut next_weight, link_point);
            } else {
                let previous_row = ranges.systematic_row_digest_row(shape, row_idx, cell_idx - 1);
                add_digest_link(
                    &mut coefficients,
                    previous_row,
                    digest_row,
                    0,
                    &mut next_weight,
                    link_point,
                );
            }
            let cell_row = ranges.cell_digest_row(shape, row_idx, cell_idx);
            add_digest_link(
                &mut coefficients,
                cell_row,
                digest_row,
                DIGEST_LEN,
                &mut next_weight,
                link_point,
            );
        }
    }

    for row_idx in 0..shape.n_rows {
        let root_row = ranges.row_root_row(row_idx);
        if row_idx == 0 {
            add_zero_digest(&mut coefficients, root_row, 0, &mut next_weight, link_point);
        } else {
            add_digest_link(
                &mut coefficients,
                ranges.row_root_row(row_idx - 1),
                root_row,
                0,
                &mut next_weight,
                link_point,
            );
        }
        let row_digest = ranges.systematic_row_digest_row(shape, row_idx, shape.num_systematic_cells() - 1);
        add_digest_link(
            &mut coefficients,
            row_digest,
            root_row,
            DIGEST_LEN,
            &mut next_weight,
            link_point,
        );
    }

    let row_tree_layers = log2_strict(shape.padded_rows());
    for cell_idx in 0..shape.num_cells() {
        for layer_idx in 0..row_tree_layers {
            let layer_width = shape.padded_rows() >> (layer_idx + 1);
            for node_idx in 0..layer_width {
                let parent_row = ranges.column_merkle_row(shape, cell_idx, layer_idx, node_idx);
                for side in 0..2 {
                    let child_idx = 2 * node_idx + side;
                    let input_offset = side * DIGEST_LEN;
                    if layer_idx == 0 {
                        if child_idx < shape.n_rows {
                            let cell_row = ranges.cell_digest_row(shape, child_idx, cell_idx);
                            add_digest_link(
                                &mut coefficients,
                                cell_row,
                                parent_row,
                                input_offset,
                                &mut next_weight,
                                link_point,
                            );
                        } else {
                            add_zero_digest(
                                &mut coefficients,
                                parent_row,
                                input_offset,
                                &mut next_weight,
                                link_point,
                            );
                        }
                    } else {
                        let child_row = ranges.column_merkle_row(shape, cell_idx, layer_idx - 1, child_idx);
                        add_digest_link(
                            &mut coefficients,
                            child_row,
                            parent_row,
                            input_offset,
                            &mut next_weight,
                            link_point,
                        );
                    }
                }
            }
        }
    }

    let column_tree_layers = log2_strict(shape.num_cells());
    for layer_idx in 0..column_tree_layers {
        let layer_width = shape.num_cells() >> (layer_idx + 1);
        for node_idx in 0..layer_width {
            let parent_row = ranges.column_root_row(shape, layer_idx, node_idx);
            for side in 0..2 {
                let child_idx = 2 * node_idx + side;
                let input_offset = side * DIGEST_LEN;
                let child_row = if layer_idx == 0 {
                    column_root_source_row(shape, ranges, child_idx)
                } else {
                    ranges.column_root_row(shape, layer_idx - 1, child_idx)
                };
                add_digest_link(
                    &mut coefficients,
                    child_row,
                    parent_row,
                    input_offset,
                    &mut next_weight,
                    link_point,
                );
            }
        }
    }

    add_digest_link(
        &mut coefficients,
        ranges.row_root_row(shape.n_rows - 1),
        ranges.final_root_row,
        0,
        &mut next_weight,
        link_point,
    );
    add_digest_link(
        &mut coefficients,
        column_commitment_source_row(shape, ranges),
        ranges.final_root_row,
        DIGEST_LEN,
        &mut next_weight,
        link_point,
    );

    merge_sparse_coefficients(&mut coefficients);
    coefficients
}

fn reserve_link_sparse_coefficients(coefficients: &mut [Vec<SparseCoeff>], shape: LeanAirShape) {
    let digest_link_estimate = shape.n_rows * (2 * shape.num_systematic_cells())
        + 2 * shape.n_rows
        + 2 * shape.num_cells() * (shape.padded_rows() - 1)
        + 2 * (shape.num_cells() - 1)
        + 2;
    let per_column = digest_link_estimate + 1;
    for column in coefficients {
        column.reserve(per_column);
    }
}

fn add_digest_link(
    coefficients: &mut [Vec<SparseCoeff>],
    source_row: usize,
    sink_row: usize,
    sink_offset: usize,
    next_weight: &mut EF,
    link_point: EF,
) {
    for limb in 0..DIGEST_LEN {
        let weight = take_link_weight(next_weight, link_point);
        add_linear_link(
            coefficients,
            TracePosition {
                row: source_row,
                column: DEDICATED_HASH_OUTPUT_START + limb,
            },
            TracePosition {
                row: sink_row,
                column: DEDICATED_HASH_INPUT_START + sink_offset + limb,
            },
            weight,
        );
    }
}

fn add_zero_digest(
    coefficients: &mut [Vec<SparseCoeff>],
    sink_row: usize,
    sink_offset: usize,
    next_weight: &mut EF,
    link_point: EF,
) {
    for limb in 0..DIGEST_LEN {
        let weight = take_link_weight(next_weight, link_point);
        add_zero_link(
            coefficients,
            TracePosition {
                row: sink_row,
                column: DEDICATED_HASH_INPUT_START + sink_offset + limb,
            },
            weight,
        );
    }
}

fn take_link_weight(next_weight: &mut EF, link_point: EF) -> EF {
    let weight = *next_weight;
    *next_weight *= link_point;
    weight
}

fn add_linear_link(coefficients: &mut [Vec<SparseCoeff>], source: TracePosition, sink: TracePosition, weight: EF) {
    let source_slot = link_column_slot(source.column);
    let sink_slot = link_column_slot(sink.column);
    add_sparse_term(coefficients, source_slot, source.row, weight);
    add_sparse_term(coefficients, sink_slot, sink.row, -weight);
}

fn add_zero_link(coefficients: &mut [Vec<SparseCoeff>], position: TracePosition, weight: EF) {
    let slot = link_column_slot(position.column);
    add_sparse_term(coefficients, slot, position.row, weight);
}

fn link_column_slot(column: usize) -> usize {
    if (DEDICATED_HASH_INPUT_START..DEDICATED_HASH_INPUT_START + DEDICATED_HASH_WIDTH).contains(&column) {
        column - DEDICATED_HASH_INPUT_START
    } else {
        assert!((DEDICATED_HASH_OUTPUT_START..DEDICATED_HASH_OUTPUT_START + DIGEST_LEN).contains(&column));
        DEDICATED_HASH_WIDTH + column - DEDICATED_HASH_OUTPUT_START
    }
}

fn column_root_source_row(shape: LeanAirShape, ranges: ScheduleRanges, cell_idx: usize) -> usize {
    if shape.padded_rows() == 1 {
        ranges.cell_digest_row(shape, 0, cell_idx)
    } else {
        ranges.column_merkle_row(shape, cell_idx, log2_strict(shape.padded_rows()) - 1, 0)
    }
}

fn column_commitment_source_row(shape: LeanAirShape, ranges: ScheduleRanges) -> usize {
    if shape.num_cells() == 1 {
        column_root_source_row(shape, ranges, 0)
    } else {
        ranges.column_root_row(shape, log2_strict(shape.num_cells()) - 1, 0)
    }
}

fn extension_basis() -> Vec<EF> {
    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    (0..dim)
        .map(|limb| EF::from_basis_coefficients_fn(|i| if i == limb { F::ONE } else { F::ZERO }))
        .collect()
}

fn constant_half_eq_eval(n_vars: usize) -> EF {
    let half = EF::from_usize(2).inverse();
    (0..n_vars).fold(EF::ONE, |acc, _| acc * half)
}

fn codeword_limb_position(shape: LeanAirShape, row_idx: usize, term_idx: usize, limb_idx: usize) -> (usize, usize) {
    let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
    let cell_idx = term_idx / shape.cell_len_ext;
    let term_in_cell = term_idx % shape.cell_len_ext;
    let limb_in_cell = term_in_cell * dim + limb_idx;

    let (chunk_idx, input_offset) = if limb_in_cell < 2 * DIGEST_LEN {
        (0, limb_in_cell)
    } else {
        let tail = limb_in_cell - 2 * DIGEST_LEN;
        (1 + tail / DIGEST_LEN, DIGEST_LEN + tail % DIGEST_LEN)
    };

    let trace_row = (row_idx * shape.num_cells() + cell_idx) * chunks_per_cell(shape) + chunk_idx;
    (trace_row, input_offset)
}

fn barycentric_slices(log_m: usize, r: EF) -> (Vec<EF>, Vec<EF>) {
    let m = 1usize << log_m;
    let w = F::two_adic_generator(log_m + 1);
    let u_inv = (w * w).inverse();
    let w_inv = w.inverse();

    let r_m = r.exp_power_of_2(log_m);
    let const_l = r_m - EF::ONE;
    let const_r = -(r_m + EF::ONE);

    let mut slice_l = Vec::with_capacity(m);
    let mut slice_r = Vec::with_capacity(m);
    let mut s_l = r;
    let mut s_r = r * w_inv;
    for _ in 0..m {
        slice_l.push(const_l / (s_l - EF::ONE));
        slice_r.push(const_r / (s_r - EF::ONE));
        s_l *= u_inv;
        s_r *= u_inv;
    }
    (slice_l, slice_r)
}

fn default_lean_air_whir_config(shape: LeanAirShape) -> WhirConfigBuilder {
    let mut config = default_whir_config(1);
    let global_n_vars = log2_ceil(DEDICATED_HASH_NUM_COLS * poseidon_rows_for_shape(shape));
    let subsequent = 5;
    if global_n_vars > config.rs_domain_initial_reduction_factor + subsequent {
        let first = 8
            .min(global_n_vars - subsequent)
            .max(config.rs_domain_initial_reduction_factor);
        let tail_n_vars = global_n_vars - first;
        config.max_num_variables_to_send_coeffs = 12.min(tail_n_vars - 1);
        config.folding_factor = FoldingFactor::new(first, subsequent);
    }
    config
}

fn powers(base: EF, len: usize) -> Vec<EF> {
    let mut out = Vec::with_capacity(len);
    let mut current = EF::ONE;
    for _ in 0..len {
        out.push(current);
        current *= base;
    }
    out
}

fn poseidon_rows_for_shape(shape: LeanAirShape) -> usize {
    (active_poseidon_rows_for_shape(shape) + 1).next_power_of_two()
}

fn active_poseidon_rows_for_shape(shape: LeanAirShape) -> usize {
    let chunks_per_cell = chunks_per_cell(shape);
    let cell_hash = shape.n_rows * shape.num_cells() * chunks_per_cell;
    let row_digest = shape.n_rows * shape.num_systematic_cells();
    let row_root = shape.n_rows;
    let column_merkle = shape.num_cells() * (shape.padded_rows() - 1);
    let column_root = shape.num_cells() - 1;
    cell_hash + row_digest + row_root + column_merkle + column_root + 1
}

fn chunks_per_cell(shape: LeanAirShape) -> usize {
    let cell_len_base = shape.cell_len_base();
    if cell_len_base <= 2 * DIGEST_LEN {
        1
    } else {
        1 + (cell_len_base - 2 * DIGEST_LEN).div_ceil(DIGEST_LEN)
    }
}

fn log2_strict(n: usize) -> usize {
    assert!(n.is_power_of_two());
    n.trailing_zeros() as usize
}

fn log2_ceil(n: usize) -> usize {
    assert!(n > 0);
    usize::BITS as usize - (n - 1).leading_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic_codewords;

    #[test]
    fn proves_and_verifies_direct_poseidon_air() {
        let shape = LeanAirShape::with_cell_len(6, 4, 8);
        let codewords = deterministic_codewords(shape);
        let proof = prove_lean_air(shape, &codewords);
        let commitment_root = proof.metadata.commitment_root;
        let verified = verify_lean_air(shape, commitment_root, proof.proof).unwrap();

        assert_eq!(verified.shape, shape);
        assert_eq!(verified.commitment_root, commitment_root);
        assert_eq!(verified.poseidon_columns, DEDICATED_HASH_NUM_COLS);
    }

    #[test]
    fn rejects_wrong_public_root() {
        let shape = LeanAirShape::with_cell_len(6, 4, 8);
        let codewords = deterministic_codewords(shape);
        let proof = prove_lean_air(shape, &codewords);
        let mut wrong_root = proof.metadata.commitment_root;
        wrong_root[0] += F::ONE;

        assert!(verify_lean_air(shape, wrong_root, proof.proof).is_err());
    }

    #[test]
    fn row_coefficient_alpha_is_bound_to_public_root_and_shape() {
        let shape = LeanAirShape::with_cell_len(6, 4, 8);
        let table = LeanAirDedicatedHashTable::build(shape, &deterministic_codewords(shape));
        let alpha = row_coefficient_alpha(shape, table.commitments.commitment_root);

        let mut other_root = table.commitments.commitment_root;
        other_root[0] += F::ONE;
        assert_ne!(alpha, row_coefficient_alpha(shape, other_root));
        assert_ne!(
            alpha,
            row_coefficient_alpha(
                LeanAirShape::with_cell_len(shape.log_m, shape.n_rows + 1, shape.cell_len_ext),
                table.commitments.commitment_root,
            )
        );
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "honest codeword parity and non-cell hash links must satisfy")
    )]
    fn rejects_non_codeword_rows() {
        let shape = LeanAirShape::with_cell_len(6, 4, 8);
        let mut codewords = deterministic_codewords(shape);
        codewords[0][0] += EF::ONE;

        let proof = prove_lean_air(shape, &codewords);
        let commitment_root = proof.metadata.commitment_root;

        assert!(verify_lean_air(shape, commitment_root, proof.proof).is_err());
    }
}
