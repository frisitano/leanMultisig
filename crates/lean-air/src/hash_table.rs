use std::any::TypeId;

use backend::{
    Air, AirBuilder, Algebra, BasedVectorSpace, EFPacking, FPacking, PrimeCharacteristicRing, SymbolicExpression,
    TwoAdicField,
};
use backend::{
    POSEIDON1_HALF_FULL_ROUNDS, POSEIDON1_PARTIAL_ROUNDS, poseidon1_final_constants, poseidon1_initial_constants,
    poseidon1_sparse_first_round_constants, poseidon1_sparse_first_row, poseidon1_sparse_m_i,
    poseidon1_sparse_scalar_round_constants, poseidon1_sparse_v,
};
use lean_vm::{DIGEST_LEN, EF, ExtraDataForBuses, F};
use rayon::prelude::*;
use tracing::info_span;

use crate::LeanAirShape;

pub const DEDICATED_HASH_WIDTH: usize = 16;
pub const DEDICATED_HASH_INPUT_START: usize = 0;
pub const DEDICATED_HASH_BEGINNING_FULL_ROUNDS_START: usize = DEDICATED_HASH_INPUT_START + DEDICATED_HASH_WIDTH;
pub const DEDICATED_HASH_PARTIAL_ROUNDS_START: usize =
    DEDICATED_HASH_BEGINNING_FULL_ROUNDS_START + DEDICATED_HASH_WIDTH * DEDICATED_HASH_HALF_INITIAL_FULL_ROUNDS;
pub const DEDICATED_HASH_ENDING_FULL_ROUNDS_START: usize =
    DEDICATED_HASH_PARTIAL_ROUNDS_START + DEDICATED_HASH_PARTIAL_ROUNDS;
pub const DEDICATED_HASH_OUTPUT_START: usize =
    DEDICATED_HASH_ENDING_FULL_ROUNDS_START + DEDICATED_HASH_WIDTH * DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS;
pub const DEDICATED_HASH_NUM_COLS: usize = DEDICATED_HASH_OUTPUT_START + DIGEST_LEN;
const DEDICATED_TRANSPOSE_COL_BLOCK: usize = 8;

const DEDICATED_HASH_HALF_INITIAL_FULL_ROUNDS: usize = POSEIDON1_HALF_FULL_ROUNDS / 2;
const DEDICATED_HASH_HALF_FINAL_FULL_ROUNDS: usize = POSEIDON1_HALF_FULL_ROUNDS / 2;
const DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS: usize = DEDICATED_HASH_HALF_FINAL_FULL_ROUNDS - 1;
const DEDICATED_HASH_PARTIAL_ROUNDS: usize = POSEIDON1_PARTIAL_ROUNDS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HashKind {
    Cell,
    SystematicRow,
    RowRoot,
    ColumnMerkle,
    ColumnRoot,
    FinalRoot,
}

impl HashKind {
    pub const COUNT: usize = 6;

    pub const fn as_index(self) -> usize {
        match self {
            Self::Cell => 0,
            Self::SystematicRow => 1,
            Self::RowRoot => 2,
            Self::ColumnMerkle => 3,
            Self::ColumnRoot => 4,
            Self::FinalRoot => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HashSchedule {
    pub shape: LeanAirShape,
    pub chunks_per_cell: usize,
    pub active_rows: usize,
    pub padded_rows: usize,
    pub counts_by_kind: [usize; HashKind::COUNT],
}

impl HashSchedule {
    pub fn new(shape: LeanAirShape) -> Self {
        let chunks_per_cell = chunks_per_cell(shape);
        let mut counts_by_kind = [0; HashKind::COUNT];
        counts_by_kind[HashKind::Cell.as_index()] = shape.n_rows * shape.num_cells() * chunks_per_cell;
        counts_by_kind[HashKind::SystematicRow.as_index()] = shape.n_rows * shape.num_systematic_cells();
        counts_by_kind[HashKind::RowRoot.as_index()] = shape.n_rows;
        counts_by_kind[HashKind::ColumnMerkle.as_index()] = shape.num_cells() * (shape.padded_rows() - 1);
        counts_by_kind[HashKind::ColumnRoot.as_index()] = shape.num_cells() - 1;
        counts_by_kind[HashKind::FinalRoot.as_index()] = 1;

        let active_rows = counts_by_kind.iter().sum();
        Self {
            shape,
            chunks_per_cell,
            active_rows,
            padded_rows: padded_table_rows(active_rows),
            counts_by_kind,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanAirCommitments {
    pub row_commitment_root: [F; DIGEST_LEN],
    pub column_commitment_root: [F; DIGEST_LEN],
    pub commitment_root: [F; DIGEST_LEN],
    pub row_digests: Vec<[F; DIGEST_LEN]>,
    pub column_roots: Vec<[F; DIGEST_LEN]>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DedicatedPoseidonCols<T> {
    inputs: [T; DEDICATED_HASH_WIDTH],
    beginning_full_rounds: [[T; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_HALF_INITIAL_FULL_ROUNDS],
    partial_rounds: [T; DEDICATED_HASH_PARTIAL_ROUNDS],
    ending_full_rounds: [[T; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS],
    outputs: [T; DIGEST_LEN],
}

#[derive(Clone, Debug)]
pub struct LeanAirDedicatedHashTable {
    pub shape: LeanAirShape,
    pub schedule: HashSchedule,
    pub commitments: LeanAirCommitments,
    pub columns: Vec<Vec<F>>,
}

impl LeanAirDedicatedHashTable {
    pub fn build(shape: LeanAirShape, codewords: &[Vec<EF>]) -> Self {
        let schedule = HashSchedule::new(shape);
        let (commitments, rows) = {
            let _cpu = system_info::enter_cpu_stage("build/dedicated_commitments");
            info_span!("lean-air build dedicated commitments")
                .in_scope(|| build_dedicated_commitments(shape, codewords))
        };
        assert_eq!(rows.len(), schedule.active_rows);

        let mut columns = {
            let _cpu = system_info::enter_cpu_stage("build/allocate_columns");
            let padding_row = dedicated_poseidon_trace([F::ZERO; 2 * DIGEST_LEN]);
            let padding_values = *dedicated_poseidon_row_values(&padding_row);
            (0..DEDICATED_HASH_NUM_COLS)
                .map(|column_idx| vec![padding_values[column_idx]; schedule.padded_rows])
                .collect::<Vec<_>>()
        };
        let _cpu = system_info::enter_cpu_stage("build/dedicated_columns");
        info_span!("lean-air dedicated columns").in_scope(|| {
            columns
                .par_chunks_mut(DEDICATED_TRANSPOSE_COL_BLOCK)
                .enumerate()
                .for_each(|(block_idx, column_block)| {
                    let column_start = block_idx * DEDICATED_TRANSPOSE_COL_BLOCK;
                    for (row_idx, row) in rows.iter().enumerate() {
                        let row_values = dedicated_poseidon_row_values(row);
                        for (local_idx, column) in column_block.iter_mut().enumerate() {
                            column[row_idx] = row_values[column_start + local_idx];
                        }
                    }
                });
        });

        Self {
            shape,
            schedule,
            commitments,
            columns,
        }
    }

    pub fn active_rows(&self) -> usize {
        self.schedule.active_rows
    }

    pub fn padded_rows(&self) -> usize {
        self.schedule.padded_rows
    }

    pub fn committed_columns(&self) -> &[Vec<F>] {
        &self.columns
    }

    pub fn output_at(&self, row: usize) -> [F; DIGEST_LEN] {
        std::array::from_fn(|i| self.columns[DEDICATED_HASH_OUTPUT_START + i][row])
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeanAirDedicatedHashAir;

impl Air for LeanAirDedicatedHashAir {
    type ExtraData = ExtraDataForBuses<EF>;

    fn degree_air(&self) -> usize {
        9
    }

    fn n_columns(&self) -> usize {
        DEDICATED_HASH_NUM_COLS
    }

    fn n_constraints(&self) -> usize {
        76
    }

    fn down_column_indexes(&self) -> Vec<usize> {
        vec![]
    }

    fn eval<AB: AirBuilder>(&self, builder: &mut AB, _extra_data: &Self::ExtraData) {
        let cols: DedicatedPoseidonCols<AB::IF> = {
            let up = builder.up();
            let (prefix, shorts, suffix) = unsafe { up.align_to::<DedicatedPoseidonCols<AB::IF>>() };
            debug_assert!(prefix.is_empty(), "Alignment should match");
            debug_assert!(suffix.is_empty(), "Alignment should match");
            debug_assert_eq!(shorts.len(), 1);
            unsafe { std::ptr::read(&shorts[0]) }
        };

        eval_dedicated_poseidon1_16(builder, &cols);
    }

    fn low_degree_air(&self) -> Option<(usize, usize)> {
        Some((3, DEDICATED_HASH_PARTIAL_ROUNDS))
    }
}

pub fn deterministic_codewords(shape: LeanAirShape) -> Vec<Vec<EF>> {
    (0..shape.n_rows)
        .into_par_iter()
        .map(|row| {
            let message = (0..shape.message_len_ext())
                .map(|col| deterministic_ext(row, col))
                .collect::<Vec<_>>();
            rs_encode(&message)
        })
        .collect()
}

fn build_dedicated_commitments(
    shape: LeanAirShape,
    codewords: &[Vec<EF>],
) -> (LeanAirCommitments, Vec<DedicatedPoseidonCols<F>>) {
    assert_eq!(codewords.len(), shape.n_rows);

    let n_cells = shape.num_cells();
    let n_systematic_cells = shape.num_systematic_cells();
    let padded_rows = shape.padded_rows();
    let schedule = HashSchedule::new(shape);

    let (mut trace_rows, mut cell_digests) = {
        let _cpu = system_info::enter_cpu_stage("build/init_trace");
        let padding_row = dedicated_poseidon_trace([F::ZERO; 2 * DIGEST_LEN]);
        (
            vec![padding_row; schedule.active_rows],
            vec![[F::ZERO; DIGEST_LEN]; n_cells * padded_rows],
        )
    };

    let cell_hash_rows = schedule.counts_by_kind[HashKind::Cell.as_index()];
    let systematic_row_rows = schedule.counts_by_kind[HashKind::SystematicRow.as_index()];
    let row_root_rows = schedule.counts_by_kind[HashKind::RowRoot.as_index()];
    let column_merkle_rows = schedule.counts_by_kind[HashKind::ColumnMerkle.as_index()];
    let column_root_rows = schedule.counts_by_kind[HashKind::ColumnRoot.as_index()];
    let final_root_rows = schedule.counts_by_kind[HashKind::FinalRoot.as_index()];

    let mut trace_offset = 0;
    let cell_trace_end = trace_offset + cell_hash_rows;
    let cell_rows_per_codeword = n_cells * schedule.chunks_per_cell;
    debug_assert_eq!(cell_hash_rows, shape.n_rows * cell_rows_per_codeword);
    let row_cell_digests = {
        let _cpu = system_info::enter_cpu_stage("build/hash_cells");
        trace_rows[trace_offset..cell_trace_end]
            .par_chunks_exact_mut(cell_rows_per_codeword)
            .zip(codewords.par_iter())
            .map(|(row_trace_rows, codeword)| {
                assert_eq!(codeword.len(), shape.codeword_len_ext());
                let row = flatten_codeword(codeword);
                let mut row_digests = vec![[F::ZERO; DIGEST_LEN]; n_cells];
                for (cell_idx, digest) in row_digests.iter_mut().enumerate() {
                    let start = cell_idx * shape.cell_len_base();
                    let cell = &row[start..start + shape.cell_len_base()];
                    let trace_start = cell_idx * schedule.chunks_per_cell;
                    *digest = hash_cell_dedicated_into(
                        cell,
                        &mut row_trace_rows[trace_start..trace_start + schedule.chunks_per_cell],
                    );
                }
                row_digests
            })
            .collect::<Vec<_>>()
    };

    {
        let _cpu = system_info::enter_cpu_stage("build/scatter_cell_digests");
        for (row_idx, row_digests) in row_cell_digests.into_iter().enumerate() {
            debug_assert_eq!(row_digests.len(), n_cells);
            for (cell_idx, digest) in row_digests.into_iter().enumerate() {
                cell_digests[cell_idx * padded_rows + row_idx] = digest;
            }
        }
    }
    trace_offset = cell_trace_end;

    let systematic_trace_end = trace_offset + systematic_row_rows;
    debug_assert_eq!(systematic_row_rows, shape.n_rows * n_systematic_cells);
    let row_digests = {
        let _cpu = system_info::enter_cpu_stage("build/systematic_row_digests");
        trace_rows[trace_offset..systematic_trace_end]
            .par_chunks_exact_mut(n_systematic_cells)
            .enumerate()
            .map(|(row_idx, row_trace_rows)| {
                chain_hash_digests_dedicated_into(
                    (0..n_systematic_cells).map(|cell_idx| cell_digests[cell_idx * padded_rows + row_idx]),
                    row_trace_rows,
                )
            })
            .collect::<Vec<_>>()
    };
    trace_offset = systematic_trace_end;

    let row_root_trace_end = trace_offset + row_root_rows;
    debug_assert_eq!(row_root_rows, shape.n_rows);
    let row_commitment_root = {
        let _cpu = system_info::enter_cpu_stage("build/row_commitment_root");
        chain_hash_digests_dedicated_into(
            row_digests.iter().copied(),
            &mut trace_rows[trace_offset..row_root_trace_end],
        )
    };
    trace_offset = row_root_trace_end;

    let column_merkle_trace_end = trace_offset + column_merkle_rows;
    debug_assert_eq!(column_merkle_rows, n_cells * (padded_rows - 1));
    let column_roots = {
        let _cpu = system_info::enter_cpu_stage("build/column_merkle_roots");
        trace_rows[trace_offset..column_merkle_trace_end]
            .par_chunks_exact_mut(padded_rows - 1)
            .enumerate()
            .map(|(cell_idx, column_trace_rows)| {
                let start = cell_idx * padded_rows;
                merkle_root_from_digests_dedicated_into(
                    cell_digests[start..start + padded_rows].to_vec(),
                    column_trace_rows,
                )
            })
            .collect::<Vec<_>>()
    };
    trace_offset = column_merkle_trace_end;

    let column_root_trace_end = trace_offset + column_root_rows;
    debug_assert_eq!(column_root_rows, n_cells - 1);
    let column_commitment_root = {
        let _cpu = system_info::enter_cpu_stage("build/column_commitment_root");
        merkle_root_from_digests_dedicated_into(
            column_roots.clone(),
            &mut trace_rows[trace_offset..column_root_trace_end],
        )
    };
    trace_offset = column_root_trace_end;

    let final_root_trace_end = trace_offset + final_root_rows;
    debug_assert_eq!(final_root_rows, 1);
    let commitment_root = {
        let _cpu = system_info::enter_cpu_stage("build/final_root");
        compress_pair_dedicated_to_row(
            row_commitment_root,
            column_commitment_root,
            &mut trace_rows[trace_offset],
        )
    };
    trace_offset = final_root_trace_end;
    debug_assert_eq!(trace_offset, schedule.active_rows);

    (
        LeanAirCommitments {
            row_commitment_root,
            column_commitment_root,
            commitment_root,
            row_digests,
            column_roots,
        },
        trace_rows,
    )
}

fn flatten_codeword(codeword: &[EF]) -> Vec<F> {
    let mut out = Vec::with_capacity(codeword.len() * <EF as BasedVectorSpace<F>>::DIMENSION);
    for value in codeword {
        out.extend_from_slice(<EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value));
    }
    out
}

fn hash_cell_dedicated_into(cell: &[F], trace_rows: &mut [DedicatedPoseidonCols<F>]) -> [F; DIGEST_LEN] {
    assert!(!cell.is_empty());
    assert!(cell.len().is_multiple_of(DIGEST_LEN));
    if cell.len() <= 2 * DIGEST_LEN {
        debug_assert_eq!(trace_rows.len(), 1);
        let mut input = [F::ZERO; 2 * DIGEST_LEN];
        input[..cell.len()].copy_from_slice(cell);
        return compress_dedicated_to_row(input, &mut trace_rows[0]);
    }

    let mut trace_idx = 0;
    let mut input = [F::ZERO; 2 * DIGEST_LEN];
    input.copy_from_slice(&cell[..2 * DIGEST_LEN]);
    let mut hash = compress_dedicated_to_row(input, &mut trace_rows[trace_idx]);
    trace_idx += 1;
    for chunk in cell[2 * DIGEST_LEN..].chunks(DIGEST_LEN) {
        let mut input = [F::ZERO; 2 * DIGEST_LEN];
        input[..DIGEST_LEN].copy_from_slice(&hash);
        input[DIGEST_LEN..DIGEST_LEN + chunk.len()].copy_from_slice(chunk);
        hash = compress_dedicated_to_row(input, &mut trace_rows[trace_idx]);
        trace_idx += 1;
    }
    debug_assert_eq!(trace_idx, trace_rows.len());
    hash
}

fn chain_hash_digests_dedicated_into(
    digests: impl IntoIterator<Item = [F; DIGEST_LEN]>,
    trace_rows: &mut [DedicatedPoseidonCols<F>],
) -> [F; DIGEST_LEN] {
    let mut state = [F::ZERO; DIGEST_LEN];
    let mut trace_idx = 0;
    for digest in digests {
        state = compress_pair_dedicated_to_row(state, digest, &mut trace_rows[trace_idx]);
        trace_idx += 1;
    }
    debug_assert_eq!(trace_idx, trace_rows.len());
    state
}

fn merkle_root_from_digests_dedicated_into(
    mut layer: Vec<[F; DIGEST_LEN]>,
    trace_rows: &mut [DedicatedPoseidonCols<F>],
) -> [F; DIGEST_LEN] {
    assert!(!layer.is_empty());
    assert!(layer.len().is_power_of_two());
    let mut trace_idx = 0;
    while layer.len() > 1 {
        let next_len = layer.len() / 2;
        let trace_level = &mut trace_rows[trace_idx..trace_idx + next_len];
        layer = layer
            .par_chunks_exact(2)
            .zip(trace_level.par_iter_mut())
            .map(|(pair, trace_row)| compress_pair_dedicated_to_row(pair[0], pair[1], trace_row))
            .collect();
        trace_idx += next_len;
    }
    debug_assert_eq!(trace_idx, trace_rows.len());
    layer[0]
}

fn compress_pair_dedicated_to_row(
    left: [F; DIGEST_LEN],
    right: [F; DIGEST_LEN],
    trace_row: &mut DedicatedPoseidonCols<F>,
) -> [F; DIGEST_LEN] {
    let mut input = [F::ZERO; 2 * DIGEST_LEN];
    input[..DIGEST_LEN].copy_from_slice(&left);
    input[DIGEST_LEN..].copy_from_slice(&right);
    compress_dedicated_to_row(input, trace_row)
}

fn compress_dedicated_to_row(input: [F; 2 * DIGEST_LEN], trace_row: &mut DedicatedPoseidonCols<F>) -> [F; DIGEST_LEN] {
    let row = dedicated_poseidon_trace(input);
    let output = row.outputs;
    *trace_row = row;
    output
}

#[inline]
fn dedicated_poseidon_row_values(cols: &DedicatedPoseidonCols<F>) -> &[F; DEDICATED_HASH_NUM_COLS] {
    debug_assert_eq!(
        std::mem::size_of::<DedicatedPoseidonCols<F>>(),
        DEDICATED_HASH_NUM_COLS * std::mem::size_of::<F>()
    );
    debug_assert_eq!(
        std::mem::align_of::<DedicatedPoseidonCols<F>>(),
        std::mem::align_of::<F>()
    );

    // DedicatedPoseidonCols is repr(C) and only contains contiguous F arrays.
    unsafe { &*(cols as *const DedicatedPoseidonCols<F> as *const [F; DEDICATED_HASH_NUM_COLS]) }
}

fn dedicated_poseidon_trace(input: [F; 2 * DIGEST_LEN]) -> DedicatedPoseidonCols<F> {
    let mut state = input;

    let mut beginning_full_rounds = [[F::ZERO; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_HALF_INITIAL_FULL_ROUNDS];
    for (round, constants) in poseidon1_initial_constants()
        .chunks_exact(2)
        .take(DEDICATED_HASH_HALF_INITIAL_FULL_ROUNDS)
        .enumerate()
    {
        generate_2_full_rounds_f(&mut state, &constants[0], &constants[1]);
        beginning_full_rounds[round] = state;
    }

    let frc = poseidon1_sparse_first_round_constants();
    for (state_i, &constant) in state.iter_mut().zip(frc.iter()) {
        *state_i += constant;
    }
    dense_mat_vec_f(poseidon1_sparse_m_i(), &mut state);

    let first_rows = poseidon1_sparse_first_row();
    let v_vecs = poseidon1_sparse_v();
    let scalar_rc = poseidon1_sparse_scalar_round_constants();
    let mut partial_rounds = [F::ZERO; DEDICATED_HASH_PARTIAL_ROUNDS];
    for round in 0..DEDICATED_HASH_PARTIAL_ROUNDS {
        state[0] = state[0].cube();
        partial_rounds[round] = state[0];
        if round < DEDICATED_HASH_PARTIAL_ROUNDS - 1 {
            state[0] += scalar_rc[round];
        }
        sparse_mat_f(&mut state, &first_rows[round], &v_vecs[round]);
    }

    let final_constants = poseidon1_final_constants();
    let mut ending_full_rounds = [[F::ZERO; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS];
    for round in 0..DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS {
        generate_2_full_rounds_f(&mut state, &final_constants[2 * round], &final_constants[2 * round + 1]);
        ending_full_rounds[round] = state;
    }

    generate_2_full_rounds_f(
        &mut state,
        &final_constants[2 * DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS],
        &final_constants[2 * DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS + 1],
    );
    for (state_i, &input_i) in state.iter_mut().zip(input.iter()) {
        *state_i += input_i;
    }
    let outputs = state[..DIGEST_LEN].try_into().unwrap();

    DedicatedPoseidonCols {
        inputs: input,
        beginning_full_rounds,
        partial_rounds,
        ending_full_rounds,
        outputs,
    }
}

fn eval_dedicated_poseidon1_16<AB: AirBuilder>(builder: &mut AB, cols: &DedicatedPoseidonCols<AB::IF>) {
    let mut state = cols.inputs;

    for (round, constants) in poseidon1_initial_constants()
        .chunks_exact(2)
        .take(DEDICATED_HASH_HALF_INITIAL_FULL_ROUNDS)
        .enumerate()
    {
        eval_2_full_rounds(
            &mut state,
            &cols.beginning_full_rounds[round],
            &constants[0],
            &constants[1],
            builder,
        );
    }

    builder.low_degree_block(&mut state, |b, state| {
        let state: &mut [AB::IF; DEDICATED_HASH_WIDTH] = state.try_into().unwrap();

        let frc = poseidon1_sparse_first_round_constants();
        for (state_i, &constant) in state.iter_mut().zip(frc.iter()) {
            add_kb(state_i, constant);
        }
        dense_mat_vec_air(poseidon1_sparse_m_i(), state);

        let first_rows = poseidon1_sparse_first_row();
        let v_vecs = poseidon1_sparse_v();
        let scalar_rc = poseidon1_sparse_scalar_round_constants();
        for round in 0..DEDICATED_HASH_PARTIAL_ROUNDS {
            state[0] = state[0].cube();
            b.assert_eq_low(state[0], cols.partial_rounds[round]);
            state[0] = cols.partial_rounds[round];
            if round < DEDICATED_HASH_PARTIAL_ROUNDS - 1 {
                add_kb(&mut state[0], scalar_rc[round]);
            }
            sparse_mat_air(state, &first_rows[round], &v_vecs[round]);
        }
    });

    let final_constants = poseidon1_final_constants();
    for round in 0..DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS {
        eval_2_full_rounds(
            &mut state,
            &cols.ending_full_rounds[round],
            &final_constants[2 * round],
            &final_constants[2 * round + 1],
            builder,
        );
    }

    for (state_i, constant) in state
        .iter_mut()
        .zip(final_constants[2 * DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS].iter())
    {
        add_kb(state_i, *constant);
        *state_i = state_i.cube();
    }
    mds_air_16(&mut state);

    for (state_i, constant) in state
        .iter_mut()
        .zip(final_constants[2 * DEDICATED_HASH_STORED_ENDING_FULL_ROUNDS + 1].iter())
    {
        add_kb(state_i, *constant);
        *state_i = state_i.cube();
    }
    mds_air_16(&mut state);

    for (state_i, input_i) in state.iter_mut().zip(cols.inputs) {
        *state_i += input_i;
    }
    for (state_i, output_i) in state.iter().zip(cols.outputs) {
        builder.assert_eq(*state_i, output_i);
    }
}

fn generate_2_full_rounds<A: PrimeCharacteristicRing + Copy + 'static>(
    state: &mut [A; DEDICATED_HASH_WIDTH],
    c1: &[F; DEDICATED_HASH_WIDTH],
    c2: &[F; DEDICATED_HASH_WIDTH],
) {
    for (state_i, constant) in state.iter_mut().zip(c1.iter()) {
        add_kb(state_i, *constant);
        *state_i = state_i.cube();
    }
    mds_air_16(state);

    for (state_i, constant) in state.iter_mut().zip(c2.iter()) {
        add_kb(state_i, *constant);
        *state_i = state_i.cube();
    }
    mds_air_16(state);
}

#[inline]
fn generate_2_full_rounds_f(
    state: &mut [F; DEDICATED_HASH_WIDTH],
    c1: &[F; DEDICATED_HASH_WIDTH],
    c2: &[F; DEDICATED_HASH_WIDTH],
) {
    for (state_i, constant) in state.iter_mut().zip(c1.iter()) {
        *state_i += *constant;
        *state_i = state_i.cube();
    }
    backend::mds_circ_16(state);

    for (state_i, constant) in state.iter_mut().zip(c2.iter()) {
        *state_i += *constant;
        *state_i = state_i.cube();
    }
    backend::mds_circ_16(state);
}

fn eval_2_full_rounds<AB: AirBuilder>(
    state: &mut [AB::IF; DEDICATED_HASH_WIDTH],
    post_full_round: &[AB::IF; DEDICATED_HASH_WIDTH],
    c1: &[F; DEDICATED_HASH_WIDTH],
    c2: &[F; DEDICATED_HASH_WIDTH],
    builder: &mut AB,
) {
    generate_2_full_rounds(state, c1, c2);
    for (state_i, post_i) in state.iter_mut().zip(post_full_round) {
        builder.assert_eq(*state_i, *post_i);
        *state_i = *post_i;
    }
}

/// Dispatch `mds_circ_16` through concrete AIR types.
/// Symbolic evaluation keeps the dense form because the circuit compiler uses it
/// to preserve dot-product precompile structure.
#[inline(always)]
fn mds_air_16<A: PrimeCharacteristicRing + 'static>(state: &mut [A; DEDICATED_HASH_WIDTH]) {
    if TypeId::of::<A>() == TypeId::of::<SymbolicExpression<F>>() {
        dense_mat_vec(mds_dense_16(), state);
        return;
    }
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                backend::mds_circ_16::<$t>(unsafe {
                    &mut *(state as *mut [A; DEDICATED_HASH_WIDTH] as *mut [$t; DEDICATED_HASH_WIDTH])
                });
                return;
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    unreachable!()
}

#[inline(always)]
fn dense_mat_vec_air<A: PrimeCharacteristicRing + Copy + 'static>(
    mat: &[[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH],
    state: &mut [A; DEDICATED_HASH_WIDTH],
) {
    if TypeId::of::<A>() == TypeId::of::<SymbolicExpression<F>>() {
        dense_mat_vec(mat, state);
        return;
    }
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                dense_mat_vec_concrete::<$t>(mat, unsafe {
                    &mut *(state as *mut [A; DEDICATED_HASH_WIDTH] as *mut [$t; DEDICATED_HASH_WIDTH])
                });
                return;
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    unreachable!()
}

fn dense_mat_vec<A: PrimeCharacteristicRing + Copy + 'static>(
    mat: &[[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH],
    state: &mut [A; DEDICATED_HASH_WIDTH],
) {
    let input = *state;
    for i in 0..DEDICATED_HASH_WIDTH {
        let mut acc = A::ZERO;
        for j in 0..DEDICATED_HASH_WIDTH {
            acc += mul_kb(input[j], mat[i][j]);
        }
        state[i] = acc;
    }
}

#[inline(always)]
fn dense_mat_vec_concrete<A>(
    mat: &[[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH],
    state: &mut [A; DEDICATED_HASH_WIDTH],
) where
    A: PrimeCharacteristicRing + Copy + std::ops::Mul<F, Output = A>,
{
    let input = *state;
    for i in 0..DEDICATED_HASH_WIDTH {
        let mut acc = A::ZERO;
        for j in 0..DEDICATED_HASH_WIDTH {
            acc += input[j] * mat[i][j];
        }
        state[i] = acc;
    }
}

#[inline]
fn dense_mat_vec_f(mat: &[[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH], state: &mut [F; DEDICATED_HASH_WIDTH]) {
    let input = *state;
    for i in 0..DEDICATED_HASH_WIDTH {
        let mut acc = F::ZERO;
        for j in 0..DEDICATED_HASH_WIDTH {
            acc += input[j] * mat[i][j];
        }
        state[i] = acc;
    }
}

#[inline(always)]
fn sparse_mat_air<A: PrimeCharacteristicRing + Copy + 'static>(
    state: &mut [A; DEDICATED_HASH_WIDTH],
    first_row: &[F; DEDICATED_HASH_WIDTH],
    v: &[F; DEDICATED_HASH_WIDTH],
) {
    if TypeId::of::<A>() == TypeId::of::<SymbolicExpression<F>>() {
        sparse_mat(state, first_row, v);
        return;
    }
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                sparse_mat_concrete::<$t>(
                    unsafe { &mut *(state as *mut [A; DEDICATED_HASH_WIDTH] as *mut [$t; DEDICATED_HASH_WIDTH]) },
                    first_row,
                    v,
                );
                return;
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    unreachable!()
}

fn sparse_mat<A: PrimeCharacteristicRing + Copy + 'static>(
    state: &mut [A; DEDICATED_HASH_WIDTH],
    first_row: &[F; DEDICATED_HASH_WIDTH],
    v: &[F; DEDICATED_HASH_WIDTH],
) {
    let old_s0 = state[0];
    let mut new_s0 = A::ZERO;
    for j in 0..DEDICATED_HASH_WIDTH {
        new_s0 += mul_kb(state[j], first_row[j]);
    }
    state[0] = new_s0;
    for i in 1..DEDICATED_HASH_WIDTH {
        state[i] += mul_kb(old_s0, v[i - 1]);
    }
}

#[inline(always)]
fn sparse_mat_concrete<A>(
    state: &mut [A; DEDICATED_HASH_WIDTH],
    first_row: &[F; DEDICATED_HASH_WIDTH],
    v: &[F; DEDICATED_HASH_WIDTH],
) where
    A: PrimeCharacteristicRing + Copy + std::ops::Mul<F, Output = A>,
{
    let old_s0 = state[0];
    let mut new_s0 = A::ZERO;
    for j in 0..DEDICATED_HASH_WIDTH {
        new_s0 += state[j] * first_row[j];
    }
    state[0] = new_s0;
    for i in 1..DEDICATED_HASH_WIDTH {
        state[i] += old_s0 * v[i - 1];
    }
}

#[inline]
fn sparse_mat_f(
    state: &mut [F; DEDICATED_HASH_WIDTH],
    first_row: &[F; DEDICATED_HASH_WIDTH],
    v: &[F; DEDICATED_HASH_WIDTH],
) {
    let old_s0 = state[0];
    let mut new_s0 = F::ZERO;
    for j in 0..DEDICATED_HASH_WIDTH {
        new_s0 += state[j] * first_row[j];
    }
    state[0] = new_s0;
    for i in 1..DEDICATED_HASH_WIDTH {
        state[i] += old_s0 * v[i - 1];
    }
}

fn mds_dense_16() -> &'static [[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH] {
    use std::sync::OnceLock;

    static MAT: OnceLock<[[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH]> = OnceLock::new();
    MAT.get_or_init(|| {
        let cols: [[F; DEDICATED_HASH_WIDTH]; DEDICATED_HASH_WIDTH] = std::array::from_fn(|j| {
            let mut e = [F::ZERO; DEDICATED_HASH_WIDTH];
            e[j] = F::ONE;
            backend::mds_circ_16(&mut e);
            e
        });
        std::array::from_fn(|i| std::array::from_fn(|j| cols[j][i]))
    })
}

#[inline(always)]
fn add_kb<A: 'static>(a: &mut A, value: F) {
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                *unsafe { &mut *(a as *mut A as *mut $t) } += value;
                return;
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    dispatch!(SymbolicExpression<F>);
    unreachable!()
}

#[inline(always)]
fn mul_kb<A: PrimeCharacteristicRing + 'static>(a: A, value: F) -> A {
    macro_rules! dispatch {
        ($t:ty) => {
            if TypeId::of::<A>() == TypeId::of::<$t>() {
                let r = unsafe { std::ptr::read(&a as *const A as *const $t) } * value;
                return unsafe { std::ptr::read(&r as *const $t as *const A) };
            }
        };
    }
    dispatch!(F);
    dispatch!(EF);
    dispatch!(FPacking<F>);
    dispatch!(EFPacking<EF>);
    dispatch!(SymbolicExpression<F>);
    unreachable!()
}

fn deterministic_ext(row: usize, col: usize) -> EF {
    EF::from_basis_coefficients_fn(|limb| F::from_usize(1 + row * 1_000_003 + col * 97 + limb))
}

fn rs_encode<A: Algebra<F> + Copy>(message: &[A]) -> Vec<A> {
    let m = message.len();
    assert!(m.is_power_of_two());
    let mut codeword = vec![A::ZERO; 2 * m];
    codeword[..m].copy_from_slice(message);
    ntt(&mut codeword);
    codeword
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

fn chunks_per_cell(shape: LeanAirShape) -> usize {
    let cell_len_base = shape.cell_len_base();
    assert!(cell_len_base.is_multiple_of(DIGEST_LEN));
    if cell_len_base <= 2 * DIGEST_LEN {
        1
    } else {
        1 + (cell_len_base - 2 * DIGEST_LEN).div_ceil(DIGEST_LEN)
    }
}

fn padded_table_rows(active_rows: usize) -> usize {
    (active_rows + 1).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedicated_hash_table_matches_native_outputs() {
        let shape = LeanAirShape::with_cell_len(4, 3, 8);
        let codewords = deterministic_codewords(shape);
        let table = LeanAirDedicatedHashTable::build(shape, &codewords);

        assert_eq!(table.padded_rows(), 128);
        assert_eq!(table.committed_columns().len(), DEDICATED_HASH_NUM_COLS);

        for row in 0..table.active_rows() {
            let input = input_at(&table, row);
            let output = table.output_at(row);
            assert_eq!(utils::poseidon16_compress(input), output);
            for limb in 0..DIGEST_LEN {
                assert_eq!(table.columns[DEDICATED_HASH_OUTPUT_START + limb][row], output[limb]);
            }
        }

        for row in table.active_rows()..table.padded_rows() {
            let input = input_at(&table, row);
            assert_eq!(input, [F::ZERO; 2 * DIGEST_LEN]);
            assert_eq!(
                table.output_at(row),
                utils::poseidon16_compress([F::ZERO; 2 * DIGEST_LEN])
            );
        }
    }

    #[test]
    fn dedicated_hash_schedule_counts_construction_4_rows() {
        let shape = LeanAirShape::with_cell_len(4, 3, 8);
        let schedule = HashSchedule::new(shape);
        let mut expected = [0; HashKind::COUNT];
        expected[HashKind::Cell.as_index()] = 48;
        expected[HashKind::SystematicRow.as_index()] = 6;
        expected[HashKind::RowRoot.as_index()] = 3;
        expected[HashKind::ColumnMerkle.as_index()] = 12;
        expected[HashKind::ColumnRoot.as_index()] = 3;
        expected[HashKind::FinalRoot.as_index()] = 1;

        assert_eq!(schedule.counts_by_kind, expected);
        assert_eq!(schedule.active_rows, 73);
    }

    #[test]
    fn poseidon_hash_root_changes_when_a_cell_changes() {
        let shape = LeanAirShape::with_cell_len(4, 3, 8);
        let mut codewords = deterministic_codewords(shape);
        let before = LeanAirDedicatedHashTable::build(shape, &codewords);
        codewords[1][0] += EF::ONE;
        let after = LeanAirDedicatedHashTable::build(shape, &codewords);

        assert_ne!(
            before.commitments.row_commitment_root,
            after.commitments.row_commitment_root
        );
        assert_ne!(
            before.commitments.column_commitment_root,
            after.commitments.column_commitment_root
        );
        assert_ne!(before.commitments.commitment_root, after.commitments.commitment_root);
    }

    #[test]
    fn dedicated_hash_air_removes_leanvm_memory_columns() {
        let air = LeanAirDedicatedHashAir;

        assert_eq!(air.n_columns(), 92);
        assert_eq!(air.n_constraints(), 76);
        assert_eq!(air.degree_air(), 9);
        assert_eq!(air.low_degree_air(), Some((3, POSEIDON1_PARTIAL_ROUNDS)));
    }

    fn input_at(table: &LeanAirDedicatedHashTable, row: usize) -> [F; 2 * DIGEST_LEN] {
        std::array::from_fn(|i| table.columns[DEDICATED_HASH_INPUT_START + i][row])
    }
}
