use backend::{BasedVectorSpace, PrimeCharacteristicRing};
use lean_vm::{EF, F, POSEIDON_PRECOMPILE_DATA};

pub const DIGEST_LEN: usize = 8;
pub const DEFAULT_CELL_LEN_EXT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeanAirShape {
    pub log_m: usize,
    pub n_rows: usize,
    pub cell_len_ext: usize,
}

impl LeanAirShape {
    pub fn new(log_m: usize, n_rows: usize) -> Self {
        Self {
            log_m,
            n_rows,
            cell_len_ext: DEFAULT_CELL_LEN_EXT,
        }
    }

    pub fn message_len_ext(self) -> usize {
        1 << self.log_m
    }

    pub fn codeword_len_ext(self) -> usize {
        2 * self.message_len_ext()
    }

    pub fn padded_rows(self) -> usize {
        self.n_rows.next_power_of_two()
    }

    pub fn num_cells(self) -> usize {
        assert!(self.codeword_len_ext().is_multiple_of(self.cell_len_ext));
        self.codeword_len_ext() / self.cell_len_ext
    }

    pub fn num_systematic_cells(self) -> usize {
        assert!(self.message_len_ext().is_multiple_of(self.cell_len_ext));
        self.message_len_ext() / self.cell_len_ext
    }

    pub fn cell_len_base(self) -> usize {
        self.cell_len_ext * <EF as BasedVectorSpace<F>>::DIMENSION
    }

    pub fn codeword_len_base(self) -> usize {
        self.codeword_len_ext() * <EF as BasedVectorSpace<F>>::DIMENSION
    }

    pub fn data_len_base(self) -> usize {
        self.n_rows * self.message_len_ext() * <EF as BasedVectorSpace<F>>::DIMENSION
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoseidonCallKind {
    CellHash,
    RowDigest,
    RowRoot,
    ColumnMerkle,
    ColumnRoot,
    FinalRoot,
}

impl PoseidonCallKind {
    pub const COUNT: usize = 6;

    pub const fn as_index(self) -> usize {
        match self {
            Self::CellHash => 0,
            Self::RowDigest => 1,
            Self::RowRoot => 2,
            Self::ColumnMerkle => 3,
            Self::ColumnRoot => 4,
            Self::FinalRoot => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowRange {
    pub start: usize,
    pub len: usize,
}

impl RowRange {
    pub fn end(self) -> usize {
        self.start + self.len
    }

    pub fn contains(self, row: usize) -> bool {
        (self.start..self.end()).contains(&row)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeanAirHashSchedule {
    pub shape: LeanAirShape,
    pub chunks_per_cell: usize,
    pub cell_hash: RowRange,
    pub row_digest: RowRange,
    pub row_root: RowRange,
    pub column_merkle: RowRange,
    pub column_root: RowRange,
    pub final_root: RowRange,
    pub padded_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeanAirHashRowMeta {
    pub kind: PoseidonCallKind,
    pub row_idx: Option<usize>,
    pub cell_idx: Option<usize>,
    pub chunk_idx: Option<usize>,
    pub layer_idx: Option<usize>,
    pub node_idx: Option<usize>,
}

impl LeanAirHashSchedule {
    pub fn new(shape: LeanAirShape) -> Self {
        assert!(shape.n_rows > 0);
        assert!(shape.cell_len_ext.is_power_of_two());
        let chunks_per_cell = chunks_per_cell(shape);
        let mut start = 0;

        let cell_hash = push_range(&mut start, shape.n_rows * shape.num_cells() * chunks_per_cell);
        let row_digest = push_range(&mut start, shape.n_rows * shape.num_systematic_cells());
        let row_root = push_range(&mut start, shape.n_rows);
        let column_merkle = push_range(&mut start, shape.num_cells() * (shape.padded_rows() - 1));
        let column_root = push_range(&mut start, shape.num_cells() - 1);
        let final_root = push_range(&mut start, 1);

        Self {
            shape,
            chunks_per_cell,
            cell_hash,
            row_digest,
            row_root,
            column_merkle,
            column_root,
            final_root,
            padded_rows: padded_table_rows(start),
        }
    }

    pub fn active_rows(self) -> usize {
        self.final_root.end()
    }

    pub fn row_meta(self, row: usize) -> Option<LeanAirHashRowMeta> {
        if self.cell_hash.contains(row) {
            let local = row - self.cell_hash.start;
            let per_row = self.shape.num_cells() * self.chunks_per_cell;
            let row_idx = local / per_row;
            let within_row = local % per_row;
            let cell_idx = within_row / self.chunks_per_cell;
            let chunk_idx = within_row % self.chunks_per_cell;
            return Some(LeanAirHashRowMeta {
                kind: PoseidonCallKind::CellHash,
                row_idx: Some(row_idx),
                cell_idx: Some(cell_idx),
                chunk_idx: Some(chunk_idx),
                layer_idx: None,
                node_idx: None,
            });
        }

        if self.row_digest.contains(row) {
            let local = row - self.row_digest.start;
            let row_idx = local / self.shape.num_systematic_cells();
            let cell_idx = local % self.shape.num_systematic_cells();
            return Some(LeanAirHashRowMeta {
                kind: PoseidonCallKind::RowDigest,
                row_idx: Some(row_idx),
                cell_idx: Some(cell_idx),
                chunk_idx: None,
                layer_idx: None,
                node_idx: None,
            });
        }

        if self.row_root.contains(row) {
            return Some(LeanAirHashRowMeta {
                kind: PoseidonCallKind::RowRoot,
                row_idx: Some(row - self.row_root.start),
                cell_idx: None,
                chunk_idx: None,
                layer_idx: None,
                node_idx: None,
            });
        }

        if self.column_merkle.contains(row) {
            let local = row - self.column_merkle.start;
            let per_column = self.shape.padded_rows() - 1;
            let cell_idx = local / per_column;
            let mut within_column = local % per_column;
            for layer_idx in 0..self.shape.padded_rows().trailing_zeros() as usize {
                let layer_width = self.shape.padded_rows() >> (layer_idx + 1);
                if within_column < layer_width {
                    return Some(LeanAirHashRowMeta {
                        kind: PoseidonCallKind::ColumnMerkle,
                        row_idx: None,
                        cell_idx: Some(cell_idx),
                        chunk_idx: None,
                        layer_idx: Some(layer_idx),
                        node_idx: Some(within_column),
                    });
                }
                within_column -= layer_width;
            }
            unreachable!("column merkle local index out of range")
        }

        if self.column_root.contains(row) {
            let mut local = row - self.column_root.start;
            for layer_idx in 0..self.shape.num_cells().trailing_zeros() as usize {
                let layer_width = self.shape.num_cells() >> (layer_idx + 1);
                if local < layer_width {
                    return Some(LeanAirHashRowMeta {
                        kind: PoseidonCallKind::ColumnRoot,
                        row_idx: None,
                        cell_idx: None,
                        chunk_idx: None,
                        layer_idx: Some(layer_idx),
                        node_idx: Some(local),
                    });
                }
                local -= layer_width;
            }
            unreachable!("column-root local index out of range")
        }

        if self.final_root.contains(row) {
            return Some(LeanAirHashRowMeta {
                kind: PoseidonCallKind::FinalRoot,
                row_idx: None,
                cell_idx: None,
                chunk_idx: None,
                layer_idx: None,
                node_idx: None,
            });
        }

        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowParitySchedule {
    pub shape: LeanAirShape,
    pub active_rows: usize,
    pub padded_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowParityRowMeta {
    pub row_idx: usize,
    pub term_idx: usize,
    pub is_first: bool,
    pub is_last: bool,
}

impl RowParitySchedule {
    pub fn new(shape: LeanAirShape) -> Self {
        let active_rows = shape.n_rows * shape.message_len_ext();
        Self {
            shape,
            active_rows,
            padded_rows: padded_table_rows(active_rows),
        }
    }

    pub fn row_meta(self, row: usize) -> Option<RowParityRowMeta> {
        if row >= self.active_rows {
            return None;
        }
        let term_idx = row % self.shape.message_len_ext();
        Some(RowParityRowMeta {
            row_idx: row / self.shape.message_len_ext(),
            term_idx,
            is_first: term_idx == 0,
            is_last: term_idx + 1 == self.shape.message_len_ext(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeanAirTablePlan {
    pub hash: LeanAirHashSchedule,
    pub row_parity: RowParitySchedule,
}

impl LeanAirTablePlan {
    pub fn new(shape: LeanAirShape) -> Self {
        Self {
            hash: LeanAirHashSchedule::new(shape),
            row_parity: RowParitySchedule::new(shape),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoseidonCompressionEvent {
    pub kind: PoseidonCallKind,
    pub input: [F; 2 * DIGEST_LEN],
    pub output: [F; DIGEST_LEN],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanAirTrace {
    pub commitments: LeanAirCommitments,
    pub poseidon_calls: Vec<PoseidonCompressionEvent>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoseidonPrecompileRequest {
    pub kind: PoseidonCallKind,
    pub precompile_data: usize,
    pub left_ptr: usize,
    pub right_ptr: usize,
    pub result_ptr: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanAirPrecompileLayout {
    pub memory: Vec<F>,
    pub poseidon_requests: Vec<PoseidonPrecompileRequest>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanAirCensus {
    pub shape: LeanAirShape,
    pub poseidon_calls_by_kind: [usize; PoseidonCallKind::COUNT],
    pub poseidon_rows: usize,
    pub poseidon_rows_padded: usize,
    pub control_rows: usize,
    pub control_rows_padded: usize,
    pub row_parity_rows: usize,
    pub row_parity_rows_padded: usize,
    pub duplicated_memory_fes: usize,
    pub shared_memory_fes_estimate: usize,
}

impl LeanAirCensus {
    pub fn poseidon_calls(&self) -> usize {
        self.poseidon_rows
    }

    pub fn max_table_rows_padded(&self) -> usize {
        self.poseidon_rows_padded
            .max(self.control_rows_padded)
            .max(self.row_parity_rows_padded)
    }

    pub fn total_active_table_rows(&self) -> usize {
        self.poseidon_rows + self.control_rows + self.row_parity_rows
    }

    pub fn total_padded_table_rows(&self) -> usize {
        self.poseidon_rows_padded + self.control_rows_padded + self.row_parity_rows_padded
    }

    pub fn fused_hash_active_table_rows(&self) -> usize {
        self.poseidon_rows + self.row_parity_rows
    }

    pub fn fused_hash_padded_table_rows(&self) -> usize {
        self.poseidon_rows_padded + self.row_parity_rows_padded
    }

    pub fn padding_overhead_bps(&self) -> usize {
        self.total_padded_table_rows() * 10_000 / self.total_active_table_rows().max(1)
    }

    pub fn fused_hash_padding_overhead_bps(&self) -> usize {
        self.fused_hash_padded_table_rows() * 10_000 / self.fused_hash_active_table_rows().max(1)
    }

    pub fn fused_hash_row_reduction_bps(&self) -> usize {
        (self.total_padded_table_rows() - self.fused_hash_padded_table_rows()) * 10_000
            / self.total_padded_table_rows().max(1)
    }

    pub fn cell_digest_count(&self) -> usize {
        self.shape.n_rows * self.shape.num_cells()
    }

    pub fn digest_binding_entries(&self) -> usize {
        self.cell_digest_count() + self.shape.n_rows * self.shape.num_systematic_cells()
    }

    pub fn materialized_rlc_parity_rows(&self) -> usize {
        self.shape.message_len_ext()
    }

    pub fn materialized_rlc_parity_rows_padded(&self) -> usize {
        padded_table_rows(self.materialized_rlc_parity_rows())
    }

    pub fn fused_hash_materialized_rlc_rows(&self) -> usize {
        self.poseidon_rows + self.materialized_rlc_parity_rows()
    }

    pub fn fused_hash_materialized_rlc_padded_rows(&self) -> usize {
        self.poseidon_rows_padded + self.materialized_rlc_parity_rows_padded()
    }

    pub fn fused_hash_virtual_rlc_padded_rows(&self) -> usize {
        self.poseidon_rows_padded
    }

    pub fn materialized_rlc_row_reduction_bps(&self) -> usize {
        (self.fused_hash_padded_table_rows() - self.fused_hash_materialized_rlc_padded_rows()) * 10_000
            / self.fused_hash_padded_table_rows().max(1)
    }

    pub fn virtual_rlc_row_reduction_bps(&self) -> usize {
        (self.fused_hash_padded_table_rows() - self.fused_hash_virtual_rlc_padded_rows()) * 10_000
            / self.fused_hash_padded_table_rows().max(1)
    }
}

/// Native Rust model of the current LeanAIR input layout.
///
/// Rows are still serialized in the leanDA circuit's evens-then-odds order:
/// `[C[0], C[2], ... C[2M-2], C[1], C[3], ... C[2M-1]]`.
/// This mirrors `lean_da_column_commit.py`; the later LeanAIR prototype can
/// swap this layout once the public protocol settles on evaluation order.
pub fn commit_codewords_evens_then_odds(shape: LeanAirShape, codewords: &[Vec<EF>]) -> LeanAirCommitments {
    build_trace_evens_then_odds(shape, codewords).commitments
}

/// Build the commitment model plus one explicit row per Poseidon16 compression.
///
/// This is the first LeanAIR witness shape: every event here can become a
/// Poseidon table row, while higher-level AIR constraints check that the event
/// selectors form the cell, row, column, and final-root chains.
pub fn build_trace_evens_then_odds(shape: LeanAirShape, codewords: &[Vec<EF>]) -> LeanAirTrace {
    assert_eq!(codewords.len(), shape.n_rows);
    assert!(shape.n_rows > 0);
    assert!(shape.cell_len_ext.is_power_of_two());

    let n_cells = shape.num_cells();
    let n_systematic_cells = shape.num_systematic_cells();
    let padded_rows = shape.padded_rows();
    let cell_len_base = shape.cell_len_base();

    let mut poseidon_calls = Vec::new();
    let mut cell_digests = vec![[F::ZERO; DIGEST_LEN]; n_cells * padded_rows];
    for (row_idx, codeword) in codewords.iter().enumerate() {
        let row = serialize_evens_then_odds(shape, codeword);
        for cell_idx in 0..n_cells {
            let start = cell_idx * cell_len_base;
            let cell = &row[start..start + cell_len_base];
            cell_digests[cell_idx * padded_rows + row_idx] =
                hash_cell(cell, PoseidonCallKind::CellHash, &mut poseidon_calls);
        }
    }

    let row_digests = (0..shape.n_rows)
        .map(|row_idx| {
            chain_hash_digests(
                (0..n_systematic_cells).map(|cell_idx| cell_digests[cell_idx * padded_rows + row_idx]),
                PoseidonCallKind::RowDigest,
                &mut poseidon_calls,
            )
        })
        .collect::<Vec<_>>();
    let row_commitment_root = chain_hash_digests(
        row_digests.iter().copied(),
        PoseidonCallKind::RowRoot,
        &mut poseidon_calls,
    );

    let column_roots = (0..n_cells)
        .map(|cell_idx| {
            let start = cell_idx * padded_rows;
            merkle_root_from_digests(
                cell_digests[start..start + padded_rows].to_vec(),
                PoseidonCallKind::ColumnMerkle,
                &mut poseidon_calls,
            )
        })
        .collect::<Vec<_>>();
    let column_commitment_root =
        merkle_root_from_digests(column_roots.clone(), PoseidonCallKind::ColumnRoot, &mut poseidon_calls);
    let commitment_root = compress_pair(
        row_commitment_root,
        column_commitment_root,
        PoseidonCallKind::FinalRoot,
        &mut poseidon_calls,
    );

    let commitments = LeanAirCommitments {
        row_commitment_root,
        column_commitment_root,
        commitment_root,
        row_digests,
        column_roots,
    };
    LeanAirTrace {
        commitments,
        poseidon_calls,
    }
}

/// Lower the LeanAIR trace to leanVM-style precompile requests.
///
/// This deliberately keeps the useful leanVM shape:
/// `Poseidon16` checks a pointer-based request through memory lookups, while a
/// custom LeanAIR request/control table will push the same bus tuples.
/// No VM execution or bytecode table is needed for this layout.
pub fn build_precompile_layout(trace: &LeanAirTrace) -> LeanAirPrecompileLayout {
    let mut memory = Vec::with_capacity(trace.poseidon_calls.len() * (3 * DIGEST_LEN));
    let mut poseidon_requests = Vec::with_capacity(trace.poseidon_calls.len());

    for call in &trace.poseidon_calls {
        let left_ptr = memory.len();
        memory.extend_from_slice(&call.input[..DIGEST_LEN]);
        let right_ptr = memory.len();
        memory.extend_from_slice(&call.input[DIGEST_LEN..]);
        let result_ptr = memory.len();
        memory.extend_from_slice(&call.output);

        poseidon_requests.push(PoseidonPrecompileRequest {
            kind: call.kind,
            precompile_data: POSEIDON_PRECOMPILE_DATA,
            left_ptr,
            right_ptr,
            result_ptr,
        });
    }

    LeanAirPrecompileLayout {
        memory,
        poseidon_requests,
    }
}

pub fn air_census(shape: LeanAirShape, trace: &LeanAirTrace) -> LeanAirCensus {
    let mut poseidon_calls_by_kind = [0; PoseidonCallKind::COUNT];
    for call in &trace.poseidon_calls {
        poseidon_calls_by_kind[call.kind.as_index()] += 1;
    }

    census_from_counts(shape, poseidon_calls_by_kind)
}

pub fn estimate_air_census(shape: LeanAirShape) -> LeanAirCensus {
    let schedule = LeanAirHashSchedule::new(shape);

    let mut poseidon_calls_by_kind = [0; PoseidonCallKind::COUNT];
    poseidon_calls_by_kind[PoseidonCallKind::CellHash.as_index()] = schedule.cell_hash.len;
    poseidon_calls_by_kind[PoseidonCallKind::RowDigest.as_index()] = schedule.row_digest.len;
    poseidon_calls_by_kind[PoseidonCallKind::RowRoot.as_index()] = schedule.row_root.len;
    poseidon_calls_by_kind[PoseidonCallKind::ColumnMerkle.as_index()] = schedule.column_merkle.len;
    poseidon_calls_by_kind[PoseidonCallKind::ColumnRoot.as_index()] = schedule.column_root.len;
    poseidon_calls_by_kind[PoseidonCallKind::FinalRoot.as_index()] = schedule.final_root.len;

    census_from_counts(shape, poseidon_calls_by_kind)
}

fn census_from_counts(shape: LeanAirShape, poseidon_calls_by_kind: [usize; PoseidonCallKind::COUNT]) -> LeanAirCensus {
    let poseidon_rows = poseidon_calls_by_kind.iter().sum();
    let control_rows = poseidon_rows;
    let row_parity_rows = shape.n_rows * shape.message_len_ext();
    let duplicated_memory_fes = poseidon_rows * 3 * DIGEST_LEN;
    let shared_memory_fes_estimate = shape.n_rows * shape.codeword_len_base() + poseidon_rows * DIGEST_LEN;

    LeanAirCensus {
        shape,
        poseidon_calls_by_kind,
        poseidon_rows,
        poseidon_rows_padded: padded_table_rows(poseidon_rows),
        control_rows,
        control_rows_padded: padded_table_rows(control_rows),
        row_parity_rows,
        row_parity_rows_padded: padded_table_rows(row_parity_rows),
        duplicated_memory_fes,
        shared_memory_fes_estimate,
    }
}

fn padded_table_rows(active_rows: usize) -> usize {
    (active_rows + 1).next_power_of_two()
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

fn push_range(start: &mut usize, len: usize) -> RowRange {
    let range = RowRange { start: *start, len };
    *start += len;
    range
}

fn serialize_evens_then_odds(shape: LeanAirShape, codeword: &[EF]) -> Vec<F> {
    let m = shape.message_len_ext();
    assert_eq!(codeword.len(), 2 * m);

    let mut out = Vec::with_capacity(shape.codeword_len_ext() * <EF as BasedVectorSpace<F>>::DIMENSION);
    for j in 0..m {
        push_ext(&mut out, &codeword[2 * j]);
    }
    for j in 0..m {
        push_ext(&mut out, &codeword[2 * j + 1]);
    }
    out
}

fn push_ext(out: &mut Vec<F>, value: &EF) {
    out.extend_from_slice(<EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value));
}

fn hash_cell(
    cell: &[F],
    kind: PoseidonCallKind,
    poseidon_calls: &mut Vec<PoseidonCompressionEvent>,
) -> [F; DIGEST_LEN] {
    assert!(!cell.is_empty());
    assert!(cell.len().is_multiple_of(DIGEST_LEN));
    if cell.len() <= 2 * DIGEST_LEN {
        let mut input = [F::ZERO; 2 * DIGEST_LEN];
        input[..cell.len()].copy_from_slice(cell);
        return compress(input, kind, poseidon_calls);
    }

    let mut input = [F::ZERO; 2 * DIGEST_LEN];
    input.copy_from_slice(&cell[..2 * DIGEST_LEN]);
    let mut hash = compress(input, kind, poseidon_calls);

    for chunk in cell[2 * DIGEST_LEN..].chunks(DIGEST_LEN) {
        let mut input = [F::ZERO; 2 * DIGEST_LEN];
        input[..DIGEST_LEN].copy_from_slice(&hash);
        input[DIGEST_LEN..DIGEST_LEN + chunk.len()].copy_from_slice(chunk);
        hash = compress(input, kind, poseidon_calls);
    }
    hash
}

fn chain_hash_digests(
    digests: impl IntoIterator<Item = [F; DIGEST_LEN]>,
    kind: PoseidonCallKind,
    poseidon_calls: &mut Vec<PoseidonCompressionEvent>,
) -> [F; DIGEST_LEN] {
    let mut state = [F::ZERO; DIGEST_LEN];
    for digest in digests {
        state = compress_pair(state, digest, kind, poseidon_calls);
    }
    state
}

fn merkle_root_from_digests(
    mut layer: Vec<[F; DIGEST_LEN]>,
    kind: PoseidonCallKind,
    poseidon_calls: &mut Vec<PoseidonCompressionEvent>,
) -> [F; DIGEST_LEN] {
    assert!(!layer.is_empty());
    assert!(layer.len().is_power_of_two());

    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|pair| compress_pair(pair[0], pair[1], kind, poseidon_calls))
            .collect();
    }
    layer[0]
}

fn compress_pair(
    left: [F; DIGEST_LEN],
    right: [F; DIGEST_LEN],
    kind: PoseidonCallKind,
    poseidon_calls: &mut Vec<PoseidonCompressionEvent>,
) -> [F; DIGEST_LEN] {
    let mut input = [F::ZERO; 2 * DIGEST_LEN];
    input[..DIGEST_LEN].copy_from_slice(&left);
    input[DIGEST_LEN..].copy_from_slice(&right);
    compress(input, kind, poseidon_calls)
}

fn compress(
    input: [F; 2 * DIGEST_LEN],
    kind: PoseidonCallKind,
    poseidon_calls: &mut Vec<PoseidonCompressionEvent>,
) -> [F; DIGEST_LEN] {
    let output = utils::poseidon16_compress(input);
    poseidon_calls.push(PoseidonCompressionEvent { kind, input, output });
    output
}
