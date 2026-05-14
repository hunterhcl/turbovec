"""Metal kernel sources for the turbovec MLX backend."""
from __future__ import annotations

import mlx.core as mx


# Apple simdgroup size; also a convenient threadgroup width for our kernels.
_TG_SIZE_DEFAULT = 32


_QUANTIZE_PACK_SOURCE = r"""
    // One threadgroup per vector. TG_SIZE threads cooperate.
    //
    // Inputs:
    //   rotated:    (n, DIM) float32 — pre-rotated unit vectors.
    //   boundaries: (N_LEVELS - 1,) float32 — Lloyd-Max boundaries.
    // Output:
    //   packed:     (n, BYTES_PER_VEC) uint8 — bit-plane layout matching
    //               turbovec/src/encode.rs::pack_codes. Plane p occupies
    //               bytes [p*PLANE_SIZE, (p+1)*PLANE_SIZE); within each
    //               plane, byte k holds coords [k*8, k*8+8), MSB-first.

    uint v = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup uchar codes_local[DIM];
    threadgroup float bnd_local[N_LEVELS - 1];

    if (tid < N_LEVELS - 1) {
        bnd_local[tid] = boundaries[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < DIM; j += TG_SIZE) {
        float r = rotated[v * DIM + j];
        uchar code = 0;
        for (int b = 0; b < N_LEVELS - 1; b++) {
            code += (r > bnd_local[b]) ? 1u : 0u;
        }
        codes_local[j] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < BYTES_PER_VEC; k += TG_SIZE) {
        uint p = k / PLANE_SIZE;
        uint bp = k % PLANE_SIZE;
        uchar byte_val = 0;
        for (uint i = 0; i < 8; i++) {
            uchar code = codes_local[bp * 8 + i];
            byte_val |= ((code >> p) & 1u) << (7 - i);
        }
        packed[v * BYTES_PER_VEC + k] = byte_val;
    }
"""


def build_quantize_pack_kernel(dim: int, bit_width: int, tg_size: int = 128):
    """Compile the fused quantize + bit-pack Metal kernel for one
    ``(dim, bit_width)``.

    Returns a callable ``(rotated, boundaries) -> packed`` where
    ``packed`` is ``(n, bit_width * dim / 8)`` ``uint8`` in the
    bit-plane layout used by ``.tv`` files.
    """
    if dim % 8 != 0:
        raise ValueError(f"dim must be a multiple of 8, got {dim}")
    if bit_width not in (2, 4):
        raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")

    n_levels = 1 << bit_width
    plane_size = dim // 8
    bytes_per_vec = bit_width * plane_size

    header = (
        f"#define DIM {dim}\n"
        f"#define BIT_WIDTH {bit_width}\n"
        f"#define N_LEVELS {n_levels}\n"
        f"#define PLANE_SIZE {plane_size}\n"
        f"#define BYTES_PER_VEC {bytes_per_vec}\n"
        f"#define TG_SIZE {tg_size}\n"
    )

    kernel = mx.fast.metal_kernel(
        name=f"turbovec_quantize_pack_d{dim}_b{bit_width}",
        input_names=["rotated", "boundaries"],
        output_names=["packed"],
        source=_QUANTIZE_PACK_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )

    def call(rotated: "mx.array", boundaries: "mx.array") -> "mx.array":
        n = rotated.shape[0]
        outputs = kernel(
            inputs=[rotated, boundaries],
            grid=(tg_size, n, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(n, bytes_per_vec)],
            output_dtypes=[mx.uint8],
        )
        return outputs[0]

    return call


_SCORE_SOURCE = r"""
    // One threadgroup per (query, vector) pair. TG_SIZE threads cooperate
    // over the inner DIM loop and tree-reduce at the end.
    //
    // Inputs:
    //   q_rot:     (nq, DIM) float32 — rotated queries.
    //   packed:    (n_db, BYTES_PER_VEC) uint8 — bit-plane codes
    //              (same layout as the encode kernel).
    //   centroids: (N_LEVELS,) float32 — Lloyd-Max centroids.
    //   norms:     (n_db,) float32 — per-vector L2 norms.
    // Output:
    //   scores:    (nq, n_db) float32 — dot products vs the reconstructed
    //              database vectors.

    uint tid = thread_position_in_threadgroup.x;
    uint v = threadgroup_position_in_grid.y;
    uint q = threadgroup_position_in_grid.z;
    uint n_db = threadgroups_per_grid.y;

    threadgroup float partial[TG_SIZE];
    threadgroup float cent_local[N_LEVELS];

    if (tid < N_LEVELS) {
        cent_local[tid] = centroids[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum = 0.0f;

    for (uint bp = tid; bp < PLANE_SIZE; bp += TG_SIZE) {
        uchar bits[BIT_WIDTH];
        for (uint p = 0; p < BIT_WIDTH; p++) {
            bits[p] = packed[v * BYTES_PER_VEC + p * PLANE_SIZE + bp];
        }
        for (uint i = 0; i < 8; i++) {
            uint j = bp * 8 + i;
            uint shift = 7 - i;
            uchar code = 0;
            for (uint p = 0; p < BIT_WIDTH; p++) {
                code |= ((bits[p] >> shift) & 1u) << p;
            }
            accum += q_rot[q * DIM + j] * cent_local[code];
        }
    }

    partial[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        scores[q * n_db + v] = partial[0] * norms[v];
    }
"""


_SCORE_BATCHED_SOURCE = r"""
    // Batched scoring: one threadgroup per (vector, query-block of QB
    // queries). TG_SIZE threads cooperate. Codes are decoded once into
    // threadgroup memory and then dot-producted against QB queries in
    // sequence, so each vector's BYTES_PER_VEC bytes are read from
    // global memory once per QB queries instead of once per query.
    //
    // Inputs:
    //   q_rot:     (NQ_PADDED, DIM) float32 — rotated queries padded
    //              to a multiple of QB with zeros.
    //   packed:    (n_db, BYTES_PER_VEC) uint8 — bit-plane codes.
    //   centroids: (N_LEVELS,) float32.
    //   norms:     (n_db,) float32.
    // Output:
    //   scores:    (NQ_PADDED, n_db) float32 — caller slices the first
    //              NQ rows.

    uint tid = thread_position_in_threadgroup.x;
    uint v = threadgroup_position_in_grid.y;
    uint q_block = threadgroup_position_in_grid.z;
    uint q_base = q_block * QB;
    uint n_db = threadgroups_per_grid.y;

    threadgroup uchar codes_local[DIM];
    threadgroup float cent_local[N_LEVELS];

    if (tid < N_LEVELS) {
        cent_local[tid] = centroids[tid];
    }

    for (uint bp = tid; bp < PLANE_SIZE; bp += TG_SIZE) {
        uchar bits[BIT_WIDTH];
        for (uint p = 0; p < BIT_WIDTH; p++) {
            bits[p] = packed[v * BYTES_PER_VEC + p * PLANE_SIZE + bp];
        }
        for (uint i = 0; i < 8; i++) {
            uchar code = 0;
            for (uint p = 0; p < BIT_WIDTH; p++) {
                code |= ((bits[p] >> (7 - i)) & 1u) << p;
            }
            codes_local[bp * 8 + i] = code;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accum[QB];
    for (uint qi = 0; qi < QB; qi++) accum[qi] = 0.0f;

    for (uint j = tid; j < DIM; j += TG_SIZE) {
        float cent_val = cent_local[codes_local[j]];
        for (uint qi = 0; qi < QB; qi++) {
            uint q = q_base + qi;
            accum[qi] += q_rot[q * DIM + j] * cent_val;
        }
    }

    // Simdgroup reduction — one instruction per query vs log2(TG_SIZE)
    // barriers through threadgroup memory. TG_SIZE must equal the
    // simdgroup width (32 on Apple GPUs) for this to be a single op.
    for (uint qi = 0; qi < QB; qi++) {
        accum[qi] = simd_sum(accum[qi]);
    }

    if (tid < QB) {
        scores[(q_base + tid) * n_db + v] = accum[tid] * norms[v];
    }
"""


def build_score_batched_kernel(
    dim: int,
    bit_width: int,
    qb: int = 4,
    tg_size: int = _TG_SIZE_DEFAULT,
):
    """Compile the query-batched scoring Metal kernel.

    Reads each vector's packed codes once per ``QB`` queries instead
    of once per query, amortizing the dominant memory cost. Caller
    must pad ``q_rot`` so ``nq_padded`` is a multiple of ``qb`` and
    slice off the padded rows from the returned scores.
    """
    if dim % 8 != 0:
        raise ValueError(f"dim must be a multiple of 8, got {dim}")
    if bit_width not in (2, 4):
        raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
    if qb < 1 or qb & (qb - 1):
        raise ValueError(f"qb must be a power of 2, got {qb}")

    n_levels = 1 << bit_width
    plane_size = dim // 8
    bytes_per_vec = bit_width * plane_size

    header = (
        f"#define DIM {dim}\n"
        f"#define BIT_WIDTH {bit_width}\n"
        f"#define N_LEVELS {n_levels}\n"
        f"#define PLANE_SIZE {plane_size}\n"
        f"#define BYTES_PER_VEC {bytes_per_vec}\n"
        f"#define TG_SIZE {tg_size}\n"
        f"#define QB {qb}\n"
    )

    kernel = mx.fast.metal_kernel(
        name=f"turbovec_score_qb{qb}_d{dim}_b{bit_width}",
        input_names=["q_rot", "packed", "centroids", "norms"],
        output_names=["scores"],
        source=_SCORE_BATCHED_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )

    def call(
        q_rot: "mx.array",
        packed: "mx.array",
        centroids: "mx.array",
        norms: "mx.array",
    ) -> "mx.array":
        nq_padded = q_rot.shape[0]
        n_db = packed.shape[0]
        if nq_padded % qb != 0:
            raise ValueError(
                f"q_rot.shape[0] ({nq_padded}) must be a multiple of qb ({qb})"
            )
        outputs = kernel(
            inputs=[q_rot, packed, centroids, norms],
            grid=(tg_size, n_db, nq_padded // qb),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(nq_padded, n_db)],
            output_dtypes=[mx.float32],
        )
        return outputs[0]

    return call


def build_score_kernel(dim: int, bit_width: int, tg_size: int = _TG_SIZE_DEFAULT):
    """Compile the dequantize-and-dot scoring Metal kernel.

    Returns a callable ``(q_rot, packed, centroids, norms) -> scores``
    where ``scores`` is shape ``(nq, n_db)`` ``float32``.
    """
    if dim % 8 != 0:
        raise ValueError(f"dim must be a multiple of 8, got {dim}")
    if bit_width not in (2, 4):
        raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")

    n_levels = 1 << bit_width
    plane_size = dim // 8
    bytes_per_vec = bit_width * plane_size

    header = (
        f"#define DIM {dim}\n"
        f"#define BIT_WIDTH {bit_width}\n"
        f"#define N_LEVELS {n_levels}\n"
        f"#define PLANE_SIZE {plane_size}\n"
        f"#define BYTES_PER_VEC {bytes_per_vec}\n"
        f"#define TG_SIZE {tg_size}\n"
    )

    kernel = mx.fast.metal_kernel(
        name=f"turbovec_score_d{dim}_b{bit_width}",
        input_names=["q_rot", "packed", "centroids", "norms"],
        output_names=["scores"],
        source=_SCORE_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )

    def call(
        q_rot: "mx.array",
        packed: "mx.array",
        centroids: "mx.array",
        norms: "mx.array",
    ) -> "mx.array":
        nq = q_rot.shape[0]
        n_db = packed.shape[0]
        outputs = kernel(
            inputs=[q_rot, packed, centroids, norms],
            grid=(tg_size, n_db, nq),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(nq, n_db)],
            output_dtypes=[mx.float32],
        )
        return outputs[0]

    return call
