use backend::BasedVectorSpace;
use lean_vm::{EF, F};

pub const DEFAULT_CELL_LEN_EXT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeanAirShape {
    pub log_m: usize,
    pub n_rows: usize,
    pub cell_len_ext: usize,
}

impl LeanAirShape {
    pub fn new(log_m: usize, n_rows: usize) -> Self {
        Self::with_cell_len(log_m, n_rows, DEFAULT_CELL_LEN_EXT)
    }

    pub fn with_cell_len(log_m: usize, n_rows: usize, cell_len_ext: usize) -> Self {
        assert!(n_rows > 0, "Construction 4 needs at least one codeword row");
        assert!(cell_len_ext > 0, "cell length must be non-zero");
        let shape = Self {
            log_m,
            n_rows,
            cell_len_ext,
        };
        assert!(
            shape.message_len_ext().is_multiple_of(cell_len_ext),
            "cell length must divide the systematic half"
        );
        assert!(
            shape.cell_len_base().is_multiple_of(lean_vm::DIGEST_LEN),
            "cell base-field limb length must be a multiple of the Poseidon digest length"
        );
        shape
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
        self.codeword_len_ext() / self.cell_len_ext
    }

    pub fn num_systematic_cells(self) -> usize {
        self.message_len_ext() / self.cell_len_ext
    }

    pub fn cell_len_base(self) -> usize {
        self.cell_len_ext * <EF as BasedVectorSpace<F>>::DIMENSION
    }

    pub fn codeword_len_base(self) -> usize {
        self.codeword_len_ext() * <EF as BasedVectorSpace<F>>::DIMENSION
    }
}
