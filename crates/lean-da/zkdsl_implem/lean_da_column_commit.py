from snark_lib import *
from barycentric import *

DIGEST_LEN = 8
LOG_LEAF_LEN_EXT = 4  # leaf size = 2^LOG_LEAF_LEN_EXT = 16 extension elements
LEAF_LEN_EXT = 2 ** LOG_LEAF_LEN_EXT
LEAF_LEN = LEAF_LEN_EXT * DIM
LEAF_NUM_CHUNKS = LEAF_LEN / DIGEST_LEN
LOG_NUM_LEAVES = LOG_M + 1 - LOG_LEAF_LEN_EXT
NUM_LEAVES = 2 ** LOG_NUM_LEAVES
LOG_NUM_SYSTEMATIC_LEAVES = LOG_M - LOG_LEAF_LEN_EXT
NUM_SYSTEMATIC_LEAVES = 2 ** LOG_NUM_SYSTEMATIC_LEAVES

N_BLOBS = N_BLOBS_PLACEHOLDER
N_BLOBS_PADDED = N_BLOBS_PADDED_PLACEHOLDER
LOG_N_BLOBS_PADDED = LOG_N_BLOBS_PADDED_PLACEHOLDER

PUB_COMMITMENT_ROOT = 0


def main():
    debug_assert(LEAF_LEN % DIGEST_LEN == 0)

    codewords = Array(N_BLOBS)
    leaf_digests = Array(NUM_LEAVES * N_BLOBS_PADDED * DIGEST_LEN)
    row_digests = Array(N_BLOBS * DIGEST_LEN)

    for row in unroll(0, N_BLOBS):
        codeword = Array(2 * M * DIM)
        hint_witness("codeword", codeword)
        codewords[row] = codeword

        for col in unroll(0, NUM_LEAVES):
            hash_leaf(
                codeword + col * LEAF_LEN,
                leaf_digests + (col * N_BLOBS_PADDED + row) * DIGEST_LEN,
            )

        hash_row_systematic_digests(leaf_digests + row * DIGEST_LEN, row_digests + row * DIGEST_LEN)

    for col in unroll(0, NUM_LEAVES):
        for row in unroll(N_BLOBS, N_BLOBS_PADDED):
            zero_digest(leaf_digests + (col * N_BLOBS_PADDED + row) * DIGEST_LEN)

    row_commitment_root = hash_row_commitment_root(row_digests)

    column_roots = Array(NUM_LEAVES * DIGEST_LEN)
    for col in unroll(0, NUM_LEAVES):
        column_root = merkle_root_from_digests(
            leaf_digests + col * N_BLOBS_PADDED * DIGEST_LEN,
            LOG_N_BLOBS_PADDED,
        )
        copy_digest(column_root, column_roots + col * DIGEST_LEN)

    column_commitment_root = merkle_root_from_digests(column_roots, LOG_NUM_LEAVES)
    commitment_root = Array(DIGEST_LEN)
    poseidon16_compress(row_commitment_root, column_commitment_root, commitment_root)
    assert_eq_digest(commitment_root, PUB_COMMITMENT_ROOT)

    r = commitment_root
    slice_L, slice_R = barycentric_slices(r)

    for row in unroll(0, N_BLOBS):
        eval_check = Array(DIM)
        dot_product_ee(codewords[row], slice_L, eval_check, M)
        dot_product_ee(codewords[row] + M * DIM, slice_R, eval_check, M)

    return


@inline
def hash_row_systematic_digests(first_row_digest, dest):
    state: Mut = Array(DIGEST_LEN)
    zero_digest(state)
    for col in unroll(0, NUM_SYSTEMATIC_LEAVES):
        new_state = Array(DIGEST_LEN)
        poseidon16_compress(state, first_row_digest + col * N_BLOBS_PADDED * DIGEST_LEN, new_state)
        state = new_state
    copy_digest(state, dest)
    return


def hash_row_commitment_root(digests):
    state: Mut = Array(DIGEST_LEN)
    zero_digest(state)
    for i in unroll(0, N_BLOBS):
        new_state = Array(DIGEST_LEN)
        poseidon16_compress(state, digests + i * DIGEST_LEN, new_state)
        state = new_state
    return state


def assert_eq_digest(a, b):
    for i in unroll(0, DIGEST_LEN):
        assert a[i] == b[i]
    return


@inline
def zero_digest(dest):
    for i in unroll(0, DIGEST_LEN):
        dest[i] = 0
    return


@inline
def copy_digest(src, dest):
    for i in unroll(0, DIGEST_LEN):
        dest[i] = src[i]
    return


@inline
def hash_leaf(leaf, dest):
    states = Array((LEAF_NUM_CHUNKS - 2) * DIGEST_LEN)
    poseidon16_compress(leaf, leaf + DIGEST_LEN, states)
    for j in unroll(1, LEAF_NUM_CHUNKS - 2):
        poseidon16_compress(
            states + (j - 1) * DIGEST_LEN,
            leaf + (j + 1) * DIGEST_LEN,
            states + j * DIGEST_LEN,
        )
    poseidon16_compress(
        states + (LEAF_NUM_CHUNKS - 3) * DIGEST_LEN,
        leaf + (LEAF_NUM_CHUNKS - 1) * DIGEST_LEN,
        dest,
    )
    return


def merkle_root_from_digests(leaves, log_num_leaves: Const):
    layer: Mut = leaves
    for k in unroll(1, log_num_leaves + 1):
        layer_size = 2 ** (log_num_leaves - k)
        new_layer = Array(layer_size * DIGEST_LEN)
        for i in unroll(0, layer_size):
            poseidon16_compress(
                layer + (2 * i) * DIGEST_LEN,
                layer + (2 * i + 1) * DIGEST_LEN,
                new_layer + i * DIGEST_LEN,
            )
        layer = new_layer

    return layer
