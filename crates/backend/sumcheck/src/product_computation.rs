use fiat_shamir::*;
use field::*;
use poly::*;
use rayon::prelude::*;
use tracing::instrument;

use crate::{SumcheckComputation, sumcheck_prove_many_rounds};

#[derive(Debug)]
pub struct ProductComputation;

impl<EF: ExtensionField<PF<EF>>> SumcheckComputation<EF> for ProductComputation {
    type ExtraData = Vec<EF>;

    fn degree(&self) -> usize {
        2
    }
    #[inline(always)]
    fn eval_base(&self, _point: &[PF<EF>], _: &Self::ExtraData) -> EF {
        unreachable!()
    }
    #[inline(always)]
    fn eval_extension(&self, point: &[EF], _: &Self::ExtraData) -> EF {
        point[0] * point[1]
    }
    #[inline(always)]
    fn eval_packed_base(&self, point: &[PFPacking<EF>], _: &Self::ExtraData) -> EFPacking<EF> {
        EFPacking::<EF>::from(point[0] * point[1])
    }
    #[inline(always)]
    fn eval_packed_extension(&self, point: &[EFPacking<EF>], _: &Self::ExtraData) -> EFPacking<EF> {
        point[0] * point[1]
    }
}

#[instrument(skip_all)]
pub fn run_product_sumcheck<EF: ExtensionField<PF<EF>>>(
    pol_a: &MleRef<'_, EF>, // evals
    pol_b: &MleRef<'_, EF>, // weights
    prover_state: &mut impl FSProver<EF>,
    mut sum: EF,
    n_rounds: usize,
    pow_bits: usize,
) -> (MultilinearPoint<EF>, EF, MleOwned<EF>, MleOwned<EF>) {
    assert!(n_rounds >= 1);
    let first_sumcheck_poly = match (pol_a, pol_b) {
        (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) => {
            if EF::DIMENSION == 5 {
                compute_product_sumcheck_polynomial_base_ext_packed::<5, _, _, _, EF>(evals, weights, sum)
            } else {
                unimplemented!()
            }
        }
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            compute_product_sumcheck_polynomial(evals, weights, sum, |e| EFPacking::<EF>::to_ext_iter([e]).collect())
        }
        (MleRef::Base(evals), MleRef::Extension(weights)) => {
            compute_product_sumcheck_polynomial(evals, weights, sum, |e| vec![e])
        }
        (MleRef::Extension(evals), MleRef::Extension(weights)) => {
            compute_product_sumcheck_polynomial(evals, weights, sum, |e| vec![e])
        }
        _ => unimplemented!(),
    };

    prover_state.add_sumcheck_polynomial(&first_sumcheck_poly.coeffs, None);
    prover_state.pow_grinding(pow_bits);
    let r1: EF = prover_state.sample();
    sum = first_sumcheck_poly.evaluate(r1);

    if n_rounds == 1 {
        return (MultilinearPoint(vec![r1]), sum, pol_a.fold(r1), pol_b.fold(r1));
    }

    let (second_sumcheck_poly, folded) = match (pol_a, pol_b) {
        (MleRef::BasePacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                });
            (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
        }
        (MleRef::ExtensionPacked(evals), MleRef::ExtensionPacked(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                });
            (second_sumcheck_poly, MleGroupOwned::ExtensionPacked(folded))
        }
        (MleRef::Base(evals), MleRef::Extension(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| vec![e]);
            (second_sumcheck_poly, MleGroupOwned::Extension(folded))
        }
        (MleRef::Extension(evals), MleRef::Extension(weights)) => {
            let (second_sumcheck_poly, folded) =
                fold_and_compute_product_sumcheck_polynomial(evals, weights, r1, sum, |e| vec![e]);
            (second_sumcheck_poly, MleGroupOwned::Extension(folded))
        }
        _ => unimplemented!(),
    };

    prover_state.add_sumcheck_polynomial(&second_sumcheck_poly.coeffs, None);
    prover_state.pow_grinding(pow_bits);
    let r2: EF = prover_state.sample();
    sum = second_sumcheck_poly.evaluate(r2);

    let original_n_vars = pol_a.n_vars();
    if let MleGroupOwned::ExtensionPacked(columns) = folded {
        if original_n_vars - n_rounds > 1 + packing_log_width::<EF>() {
            let mut columns = columns.into_iter();
            let evals = columns.next().unwrap();
            let weights = columns.next().unwrap();
            debug_assert!(columns.next().is_none());
            let (mut challenges, sum, pol_a, pol_b) = continue_product_sumcheck_extension_packed_in_place(
                evals,
                weights,
                prover_state,
                sum,
                r2,
                n_rounds - 2,
                pow_bits,
            );
            challenges.0.splice(0..0, [r1, r2]);
            return (challenges, sum, pol_a, pol_b);
        }
        let folded = MleGroupOwned::ExtensionPacked(columns);
        let (mut challenges, folds, sum) = sumcheck_prove_many_rounds(
            folded,
            Some(r2),
            &ProductComputation {},
            &vec![],
            None,
            prover_state,
            sum,
            None,
            n_rounds - 2,
            false,
            pow_bits,
        );

        challenges.splice(0..0, [r1, r2]);
        let [pol_a, pol_b] = folds.split().try_into().unwrap();
        return (challenges, sum, pol_a, pol_b);
    }

    let (mut challenges, folds, sum) = sumcheck_prove_many_rounds(
        folded,
        Some(r2),
        &ProductComputation {},
        &vec![],
        None,
        prover_state,
        sum,
        None,
        n_rounds - 2,
        false,
        pow_bits,
    );

    challenges.splice(0..0, [r1, r2]);
    let [pol_a, pol_b] = folds.split().try_into().unwrap();
    (challenges, sum, pol_a, pol_b)
}

#[allow(clippy::too_many_arguments)]
fn continue_product_sumcheck_extension_packed_in_place<EF: ExtensionField<PF<EF>>>(
    mut evals: Vec<EFPacking<EF>>,
    mut weights: Vec<EFPacking<EF>>,
    prover_state: &mut impl FSProver<EF>,
    mut sum: EF,
    mut pending_challenge: EF,
    n_rounds: usize,
    pow_bits: usize,
) -> (MultilinearPoint<EF>, EF, MleOwned<EF>, MleOwned<EF>) {
    let mut challenges = Vec::with_capacity(n_rounds);
    for _ in 0..n_rounds {
        let poly = fold_and_compute_product_sumcheck_polynomial_extension_packed_in_place(
            &mut evals,
            &mut weights,
            pending_challenge,
            sum,
        );
        prover_state.add_sumcheck_polynomial(&poly.coeffs, None);
        prover_state.pow_grinding(pow_bits);
        pending_challenge = prover_state.sample();
        sum = poly.evaluate(pending_challenge);
        challenges.push(pending_challenge);
    }

    fold_product_extension_packed_in_place(&mut evals, &mut weights, pending_challenge);
    (
        MultilinearPoint(challenges),
        sum,
        MleOwned::ExtensionPacked(evals),
        MleOwned::ExtensionPacked(weights),
    )
}

pub fn compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync,
    EF: Field,
    EFPacking: Algebra<F> + Copy + Send + Sync,
>(
    pol_0: &[F],         // evals
    pol_1: &[EFPacking], // weights
    sum: EF,
    decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> DensePolynomial<EF> {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());

    let num_elements = n;

    let (c0_packed, c2_packed) = if num_elements < PARALLEL_THRESHOLD {
        pol_0[..n / 2]
            .iter()
            .zip(pol_0[n / 2..].iter())
            .zip(pol_1[..n / 2].iter().zip(pol_1[n / 2..].iter()))
            .map(sumcheck_quadratic)
            .fold((EFPacking::ZERO, EFPacking::ZERO), |(a0, a2), (b0, b2)| {
                (a0 + b0, a2 + b2)
            })
    } else {
        pol_0[..n / 2]
            .par_iter()
            .zip(pol_0[n / 2..].par_iter())
            .zip(pol_1[..n / 2].par_iter().zip(pol_1[n / 2..].par_iter()))
            .map(sumcheck_quadratic)
            .reduce(
                || (EFPacking::ZERO, EFPacking::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            )
    };

    let c0 = decompose(c0_packed).into_iter().sum::<EF>();
    let c2 = decompose(c2_packed).into_iter().sum::<EF>();
    let c1 = sum - c0.double() - c2;

    DensePolynomial::new(vec![c0, c1, c2])
}

// using delayed modular reduction
pub fn compute_product_sumcheck_polynomial_base_ext_packed<
    const DIM: usize,
    F: PrimeField32,
    PF: PackedField<Scalar = F>,
    EFP: BasedVectorSpace<PF> + Copy + Send + Sync,
    EF: Field + BasedVectorSpace<F>,
>(
    pol_0: &[PF],
    pol_1: &[EFP],
    sum: EF,
) -> DensePolynomial<EF> {
    assert_eq!(DIM, EF::DIMENSION);
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());
    let half = n / 2;

    type Acc<const D: usize> = ([u128; D], [i128; D]);

    let chunk_size = 1024;

    let (c0_acc, c2_acc) = pol_0[..half]
        .par_chunks(chunk_size)
        .zip(pol_0[half..].par_chunks(chunk_size))
        .zip(
            pol_1[..half]
                .par_chunks(chunk_size)
                .zip(pol_1[half..].par_chunks(chunk_size)),
        )
        .map(|((b_lo, b_hi), (e_lo, e_hi))| {
            let mut c0 = [0u128; DIM];
            let mut c2 = [0i128; DIM];
            for i in 0..b_lo.len() {
                let x0_lanes = b_lo[i].as_slice();
                let x1_lanes = b_hi[i].as_slice();
                let y0_coords = e_lo[i].as_basis_coefficients_slice();
                let y1_coords = e_hi[i].as_basis_coefficients_slice();
                for j in 0..DIM {
                    let y0_j = y0_coords[j].as_slice();
                    let y1_j = y1_coords[j].as_slice();
                    for lane in 0..PF::WIDTH {
                        let x0 = x0_lanes[lane].to_unique_u32() as u64;
                        let y0 = y0_j[lane].to_unique_u32();
                        let y1 = y1_j[lane].to_unique_u32();
                        c0[j] += (y0 as u64 * x0) as u128;
                        c2[j] += (y1 as i64 - y0 as i64) as i128
                            * (x1_lanes[lane].to_unique_u32() as i64 - x0 as i64) as i128;
                    }
                }
            }
            (c0, c2)
        })
        .reduce(
            || ([0u128; DIM], [0i128; DIM]),
            |(mut a0, mut a2): Acc<DIM>, (b0, b2): Acc<DIM>| {
                for j in 0..DIM {
                    a0[j] += b0[j];
                    a2[j] += b2[j];
                }
                (a0, a2)
            },
        );

    let c0 = EF::from_basis_coefficients_fn(|j| F::reduce_product_sum(c0_acc[j]));
    let c2 = EF::from_basis_coefficients_fn(|j| F::reduce_signed_product_sum(c2_acc[j]));
    let c1 = sum - c0.double() - c2;

    DensePolynomial::new(vec![c0, c1, c2])
}

pub fn fold_and_compute_product_sumcheck_polynomial<
    F: PrimeCharacteristicRing + Copy + Send + Sync + 'static,
    EF: Field,
    EFPacking: Algebra<F> + From<EF> + Copy + Send + Sync + 'static,
>(
    pol_0: &[F],         // evals
    pol_1: &[EFPacking], // weights
    prev_folding_factor: EF,
    sum: EF,
    decompose: impl Fn(EFPacking) -> Vec<EF>,
) -> (DensePolynomial<EF>, Vec<Vec<EFPacking>>) {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());
    let prev_folding_factor_packed = EFPacking::from(prev_folding_factor);

    let mut pol_0_folded = unsafe { uninitialized_vec::<EFPacking>(n / 2) };
    let mut pol_1_folded = unsafe { uninitialized_vec::<EFPacking>(n / 2) };

    #[allow(clippy::type_complexity)]
    let process_element = |(p0_prev, p0_f): (((&F, &F), (&F, &F)), (&mut EFPacking, &mut EFPacking)),
                           (p1_prev, p1_f): (
        ((&EFPacking, &EFPacking), (&EFPacking, &EFPacking)),
        (&mut EFPacking, &mut EFPacking),
    )| {
        let diff_0 = *p0_prev.1.0 - *p0_prev.0.0;
        let diff_1 = *p0_prev.1.1 - *p0_prev.0.1;
        let x_0 = prev_folding_factor_packed * diff_0 + *p0_prev.0.0;
        let x_1 = prev_folding_factor_packed * diff_1 + *p0_prev.0.1;
        *p0_f.0 = x_0;
        *p0_f.1 = x_1;

        let y_0 = prev_folding_factor_packed * (*p1_prev.1.0 - *p1_prev.0.0) + *p1_prev.0.0;
        let y_1 = prev_folding_factor_packed * (*p1_prev.1.1 - *p1_prev.0.1) + *p1_prev.0.1;
        *p1_f.0 = y_0;
        *p1_f.1 = y_1;

        sumcheck_quadratic(((&x_0, &x_1), (&y_0, &y_1)))
    };

    let (c0_packed, c2_packed) = if n < PARALLEL_THRESHOLD {
        zip_fold_2(pol_0, &mut pol_0_folded)
            .zip(zip_fold_2(pol_1, &mut pol_1_folded))
            .map(|(p0, p1)| process_element(p0, p1))
            .fold((EFPacking::ZERO, EFPacking::ZERO), |(a0, a2), (b0, b2)| {
                (a0 + b0, a2 + b2)
            })
    } else {
        par_zip_fold_2(pol_0, &mut pol_0_folded)
            .zip(par_zip_fold_2(pol_1, &mut pol_1_folded))
            .map(|(p0, p1)| process_element(p0, p1))
            .reduce(
                || (EFPacking::ZERO, EFPacking::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            )
    };

    let c0 = decompose(c0_packed).into_iter().sum::<EF>();
    let c2 = decompose(c2_packed).into_iter().sum::<EF>();
    let c1 = sum - c0.double() - c2;

    (DensePolynomial::new(vec![c0, c1, c2]), vec![pol_0_folded, pol_1_folded])
}

fn fold_and_compute_product_sumcheck_polynomial_extension_packed_in_place<EF: ExtensionField<PF<EF>>>(
    pol_0: &mut Vec<EFPacking<EF>>,
    pol_1: &mut Vec<EFPacking<EF>>,
    prev_folding_factor: EF,
    sum: EF,
) -> DensePolynomial<EF> {
    let (c0_packed, c2_packed) = fold_product_extension_packed_in_place_core(pol_0, pol_1, prev_folding_factor, true);
    let c0 = EFPacking::<EF>::to_ext_iter([c0_packed]).sum::<EF>();
    let c2 = EFPacking::<EF>::to_ext_iter([c2_packed]).sum::<EF>();
    let c1 = sum - c0.double() - c2;

    DensePolynomial::new(vec![c0, c1, c2])
}

fn fold_product_extension_packed_in_place<EF: ExtensionField<PF<EF>>>(
    pol_0: &mut Vec<EFPacking<EF>>,
    pol_1: &mut Vec<EFPacking<EF>>,
    prev_folding_factor: EF,
) {
    let _ = fold_product_extension_packed_in_place_core(pol_0, pol_1, prev_folding_factor, false);
}

fn fold_product_extension_packed_in_place_core<EF: ExtensionField<PF<EF>>>(
    pol_0: &mut Vec<EFPacking<EF>>,
    pol_1: &mut Vec<EFPacking<EF>>,
    prev_folding_factor: EF,
    compute_poly: bool,
) -> (EFPacking<EF>, EFPacking<EF>) {
    let n = pol_0.len();
    assert_eq!(n, pol_1.len());
    assert!(n.is_power_of_two());
    assert!(n.is_multiple_of(4));

    let half = n / 2;
    let quarter = n / 4;
    let prev_folding_factor_packed = EFPacking::<EF>::from(prev_folding_factor);

    let (p0_low, p0_high) = pol_0.split_at_mut(half);
    let (p0_ll, p0_lr) = p0_low.split_at_mut(quarter);
    let (p0_rl, p0_rr) = p0_high.split_at(quarter);

    let (p1_low, p1_high) = pol_1.split_at_mut(half);
    let (p1_ll, p1_lr) = p1_low.split_at_mut(quarter);
    let (p1_rl, p1_rr) = p1_high.split_at(quarter);

    let fold_chunk = |p0_ll: &mut [EFPacking<EF>],
                      p0_lr: &mut [EFPacking<EF>],
                      p0_rl: &[EFPacking<EF>],
                      p0_rr: &[EFPacking<EF>],
                      p1_ll: &mut [EFPacking<EF>],
                      p1_lr: &mut [EFPacking<EF>],
                      p1_rl: &[EFPacking<EF>],
                      p1_rr: &[EFPacking<EF>]|
     -> (EFPacking<EF>, EFPacking<EF>) {
        let mut c0 = EFPacking::<EF>::ZERO;
        let mut c2 = EFPacking::<EF>::ZERO;
        for i in 0..p0_ll.len() {
            let x_ll = p0_ll[i];
            let x_lr = p0_lr[i];
            let x_0 = prev_folding_factor_packed * (p0_rl[i] - x_ll) + x_ll;
            let x_1 = prev_folding_factor_packed * (p0_rr[i] - x_lr) + x_lr;
            p0_ll[i] = x_0;
            p0_lr[i] = x_1;

            let y_ll = p1_ll[i];
            let y_lr = p1_lr[i];
            let y_0 = prev_folding_factor_packed * (p1_rl[i] - y_ll) + y_ll;
            let y_1 = prev_folding_factor_packed * (p1_rr[i] - y_lr) + y_lr;
            p1_ll[i] = y_0;
            p1_lr[i] = y_1;

            if compute_poly {
                let (local_c0, local_c2) = sumcheck_quadratic(((&x_0, &x_1), (&y_0, &y_1)));
                c0 += local_c0;
                c2 += local_c2;
            }
        }
        (c0, c2)
    };

    let (c0_packed, c2_packed) = if quarter < PARALLEL_THRESHOLD {
        fold_chunk(p0_ll, p0_lr, p0_rl, p0_rr, p1_ll, p1_lr, p1_rl, p1_rr)
    } else {
        let chunk_size = 1024;
        p0_ll
            .par_chunks_mut(chunk_size)
            .zip(p0_lr.par_chunks_mut(chunk_size))
            .zip(p0_rl.par_chunks(chunk_size))
            .zip(p0_rr.par_chunks(chunk_size))
            .zip(p1_ll.par_chunks_mut(chunk_size))
            .zip(p1_lr.par_chunks_mut(chunk_size))
            .zip(p1_rl.par_chunks(chunk_size))
            .zip(p1_rr.par_chunks(chunk_size))
            .map(|(((((((p0_ll, p0_lr), p0_rl), p0_rr), p1_ll), p1_lr), p1_rl), p1_rr)| {
                fold_chunk(p0_ll, p0_lr, p0_rl, p0_rr, p1_ll, p1_lr, p1_rl, p1_rr)
            })
            .reduce(
                || (EFPacking::<EF>::ZERO, EFPacking::<EF>::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            )
    };

    pol_0.truncate(half);
    pol_1.truncate(half);
    (c0_packed, c2_packed)
}

#[inline(always)]
pub fn sumcheck_quadratic<F, EF>(((&x_0, &x_1), (&y_0, &y_1)): ((&F, &F), (&EF, &EF))) -> (EF, EF)
where
    F: PrimeCharacteristicRing + Copy,
    EF: Algebra<F> + Copy,
{
    let constant = y_0 * x_0;
    let quadratic = (y_1 - y_0) * (x_1 - x_0);
    (constant, quadratic)
}
