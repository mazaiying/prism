#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

namespace prism {

constexpr uint32_t PRED_COL0_RANGE = 1u;
constexpr uint32_t PRED_COL1_RANGE = 2u;
constexpr uint32_t PRED_COL2_CAT   = 4u;
constexpr uint32_t PRED_ALG_EXPR   = 8u;

struct alignas(32) TileHeader {
    uint16_t c0_min, c0_max;
    uint16_t c1_min, c1_max;
    uint16_t c3_min, c3_max;
    uint16_t valid_cnt;
    uint16_t flags;
    uint64_t cat_bloom64;
    uint32_t row_offset;
    int32_t  cluster_id;
};
static_assert(sizeof(TileHeader) == 32, "TileHeader must be exactly 32 bytes");

__device__ __forceinline__ __half double_to_half_ru_pos(double val_d) {
    if (val_d <= 0.0) return __ushort_as_half(0x0000u);
    float vf = __double2float_ru(val_d);
    __half h = __float2half_rn(vf);
    float hf = __half2float(h);
    if (hf < vf) {
        unsigned short bits = __half_as_ushort(h);
        if (bits < 0x7C00u) bits += 1u;
        h = __ushort_as_half(bits);
    }
    return h;
}

__device__ __forceinline__ int check_tile_header(
    const TileHeader& th,
    uint32_t op_flags,
    int32_t l0, int32_t r0,
    int32_t l1, int32_t r1,
    uint64_t cat_mask,
    int32_t w0, int32_t w3, int32_t alg_bound
) {
    if (th.valid_cnt == 0) return 0; // ALL_DEAD
    bool all_hit = ((th.flags & 1u) == 0u); // Bit 0 of flags marks tombstones present in tile

    if (op_flags & PRED_COL0_RANGE) {
        if ((int32_t)th.c0_max < l0 || (int32_t)th.c0_min > r0) return 0;
        if ((int32_t)th.c0_min < l0 || (int32_t)th.c0_max > r0) all_hit = false;
    }
    if (op_flags & PRED_COL1_RANGE) {
        if ((int32_t)th.c1_max < l1 || (int32_t)th.c1_min > r1) return 0;
        if ((int32_t)th.c1_min < l1 || (int32_t)th.c1_max > r1) all_hit = false;
    }
    if (op_flags & PRED_COL2_CAT) {
        if ((th.cat_bloom64 & cat_mask) == 0ULL) return 0;
        if ((th.cat_bloom64 & (~cat_mask)) != 0ULL) all_hit = false;
    }
    if (op_flags & PRED_ALG_EXPR) {
        // w0 >= 0, w3 >= 0 in our SQL algebraic predicates
        int64_t min_v = (int64_t)w0 * (int64_t)th.c0_min + (int64_t)w3 * (int64_t)th.c3_min;
        int64_t max_v = (int64_t)w0 * (int64_t)th.c0_max + (int64_t)w3 * (int64_t)th.c3_max;
        if (min_v > (int64_t)alg_bound) return 0;
        if (max_v > (int64_t)alg_bound) all_hit = false;
    }
    return all_hit ? 2 : 1; // 2 = ALL_HIT, 1 = BOUNDARY
}

__device__ __forceinline__ bool eval_slap_tuple(
    uint64_t packed,
    uint32_t op_flags,
    int32_t l0, int32_t r0,
    int32_t l1, int32_t r1,
    uint64_t cat_mask,
    int32_t w0, int32_t w3, int32_t alg_bound
) {
    if ((packed >> 63) != 0ULL) return false; // Tombstone bit for O(1) incremental deletion
    int32_t c0 = (int32_t)(packed & 0xFFFFULL);
    int32_t c1 = (int32_t)((packed >> 16) & 0xFFFFULL);
    int32_t c3 = (int32_t)((packed >> 32) & 0xFFFFULL);
    uint32_t c2 = (uint32_t)((packed >> 48) & 0x3FULL);

    bool ok = true;
    if (op_flags & PRED_COL0_RANGE) {
        ok = ok && (c0 >= l0 && c0 <= r0);
    }
    if (op_flags & PRED_COL1_RANGE) {
        ok = ok && (c1 >= l1 && c1 <= r1);
    }
    if (op_flags & PRED_COL2_CAT) {
        ok = ok && (((1ULL << c2) & cat_mask) != 0ULL);
    }
    if (op_flags & PRED_ALG_EXPR) {
        ok = ok && ((w0 * c0 + w3 * c3) <= alg_bound);
    }
    return ok;
}

template <typename VecT>
__global__ void encode_prism_tiles_kernel(
    const VecT* __restrict__ vecs_chunk,        // [n_tiles * 32, D]
    const uint16_t* __restrict__ c0_chunk,      // [n_tiles * 32]
    const uint16_t* __restrict__ c1_chunk,
    const uint16_t* __restrict__ c2_chunk,
    const uint16_t* __restrict__ c3_chunk,
    const uint16_t* __restrict__ tile_valid,    // [n_tiles]
    const int32_t* __restrict__ tile_clusters,  // [n_tiles]
    const int32_t* __restrict__ target_tiles,   // nullptr (contiguous from base_tile_id) or [n_tiles]
    const float* __restrict__ cluster_o,        // [nlist, D_pad]
    const float* __restrict__ cluster_delta,    // [nlist, D_pad]
    const int32_t* __restrict__ block_bounds,   // [M + 1] in units of 32-dim chunks
    int32_t base_tile_id,
    int32_t n_tiles,
    int64_t N_pad,
    int32_t D,
    int32_t D_pad,
    int32_t M,
    bool use_sign_bit,
    TileHeader* __restrict__ out_headers,       // [N_wtiles]
    uint64_t* __restrict__ out_slap_cols,       // [N_pad]
    uint4* __restrict__ out_int4_codes,         // [chunks32, N_pad]
    uint32_t* __restrict__ out_sign_bits,       // [chunks32, N_pad]
    __half2* __restrict__ out_ball_meta         // [M, N_pad]
) {
    int loc_t = blockIdx.x;
    if (loc_t >= n_tiles) return;
    int lane = threadIdx.x; // 0..31

    int64_t global_tile = (target_tiles != nullptr) ? (int64_t)target_tiles[loc_t] : ((int64_t)base_tile_id + loc_t);
    int64_t global_row  = global_tile * 32 + lane;
    int64_t chunk_row   = (int64_t)loc_t * 32 + lane;

    uint16_t vcnt = tile_valid[loc_t];
    int32_t cid   = tile_clusters[loc_t];
    bool is_valid = (lane < (int)vcnt);

    uint16_t v0 = is_valid ? c0_chunk[chunk_row] : 0;
    uint16_t v1 = is_valid ? c1_chunk[chunk_row] : 0;
    uint16_t v2 = is_valid ? (uint16_t)(c2_chunk[chunk_row] & 63u) : 0;
    uint16_t v3 = is_valid ? c3_chunk[chunk_row] : 0;

    out_slap_cols[global_row] =
        (uint64_t)v0 |
        ((uint64_t)v1 << 16) |
        ((uint64_t)v3 << 32) |
        ((uint64_t)v2 << 48);

    // Warp-reduce TileHeader statistics across valid lanes
    uint16_t mn0 = is_valid ? v0 : 65535u;
    uint16_t mx0 = is_valid ? v0 : 0u;
    uint16_t mn1 = is_valid ? v1 : 65535u;
    uint16_t mx1 = is_valid ? v1 : 0u;
    uint16_t mn3 = is_valid ? v3 : 65535u;
    uint16_t mx3 = is_valid ? v3 : 0u;
    uint64_t blm = is_valid ? (1ULL << v2) : 0ULL;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        uint16_t o_mn0 = __shfl_xor_sync(0xffffffffu, mn0, off);
        uint16_t o_mx0 = __shfl_xor_sync(0xffffffffu, mx0, off);
        uint16_t o_mn1 = __shfl_xor_sync(0xffffffffu, mn1, off);
        uint16_t o_mx1 = __shfl_xor_sync(0xffffffffu, mx1, off);
        uint16_t o_mn3 = __shfl_xor_sync(0xffffffffu, mn3, off);
        uint16_t o_mx3 = __shfl_xor_sync(0xffffffffu, mx3, off);
        uint64_t o_blm = __shfl_xor_sync(0xffffffffu, blm, off);
        mn0 = mn0 < o_mn0 ? mn0 : o_mn0;
        mx0 = mx0 > o_mx0 ? mx0 : o_mx0;
        mn1 = mn1 < o_mn1 ? mn1 : o_mn1;
        mx1 = mx1 > o_mx1 ? mx1 : o_mx1;
        mn3 = mn3 < o_mn3 ? mn3 : o_mn3;
        mx3 = mx3 > o_mx3 ? mx3 : o_mx3;
        blm |= o_blm;
    }

    if (lane == 0) {
        TileHeader th;
        th.c0_min = vcnt > 0 ? mn0 : 65535u;
        th.c0_max = vcnt > 0 ? mx0 : 0u;
        th.c1_min = vcnt > 0 ? mn1 : 65535u;
        th.c1_max = vcnt > 0 ? mx1 : 0u;
        th.c3_min = vcnt > 0 ? mn3 : 65535u;
        th.c3_max = vcnt > 0 ? mx3 : 0u;
        th.valid_cnt = vcnt;
        th.flags = 0u;
        th.cat_bloom64 = blm;
        th.row_offset = (uint32_t)(global_tile * 32);
        th.cluster_id = cid;
        out_headers[global_tile] = th;
    }

    const float* o_ptr = cluster_o + (int64_t)cid * D_pad;
    const float* d_ptr = cluster_delta + (int64_t)cid * D_pad;
    const VecT* x_ptr  = vecs_chunk + chunk_row * D;

    for (int m = 0; m < M; ++m) {
        int u_beg = block_bounds[m];
        int u_end = block_bounds[m + 1];
        int blk_dims = 0;
        float l1_sum = 0.0f;

        // Pass A: Quantize INT4 + 1-bit residual sign and accumulate L1 residual magnitude
        for (int u = u_beg; u < u_end; ++u) {
            uint32_t words[4] = {0u, 0u, 0u, 0u};
            uint32_t s_word = 0u;

            #pragma unroll
            for (int d_loc = 0; d_loc < 32; ++d_loc) {
                int j = u * 32 + d_loc;
                if (j < D && is_valid) {
                    float xj = (float)x_ptr[j];
                    float oj = o_ptr[j];
                    float dj = d_ptr[j];
                    float inv_d = (dj > 1e-12f) ? (1.0f / dj) : 0.0f;
                    int q4 = __float2int_rn((xj - oj) * inv_d);
                    q4 = q4 < 0 ? 0 : (q4 > 15 ? 15 : q4);

                    float x_hat = fmaf(dj, (float)q4, oj);
                    float res = xj - x_hat;
                    uint32_t sb = (res >= 0.0f) ? 1u : 0u;
                    l1_sum += fabsf(res) * inv_d;
                    blk_dims += 1;

                    words[d_loc >> 3] |= ((uint32_t)q4 << ((d_loc & 7) * 4));
                    s_word |= (sb << d_loc);
                }
            }
            uint4 pack4 = make_uint4(words[0], words[1], words[2], words[3]);
            out_int4_codes[(int64_t)u * N_pad + global_row] = pack4;
            out_sign_bits[(int64_t)u * N_pad + global_row]  = s_word;
        }

        __half gamma_h = (use_sign_bit && blk_dims > 0)
            ? __float2half_rn(l1_sum / (float)blk_dims)
            : __ushort_as_half(0x0000u);
        float gamma_f = __half2float(gamma_h);

        // Pass B: Exact device-identical float32 reconstruction + FP64 Euclidean residual norm rounded UP
        double err2 = 0.0;
        for (int u = u_beg; u < u_end; ++u) {
            uint4 pack4 = out_int4_codes[(int64_t)u * N_pad + global_row];
            uint32_t s_word = out_sign_bits[(int64_t)u * N_pad + global_row];
            uint32_t words[4] = {pack4.x, pack4.y, pack4.z, pack4.w};

            #pragma unroll
            for (int d_loc = 0; d_loc < 32; ++d_loc) {
                int j = u * 32 + d_loc;
                if (j < D && is_valid) {
                    uint32_t q4 = (words[d_loc >> 3] >> ((d_loc & 7) * 4)) & 0xFu;
                    uint32_t sb = (s_word >> d_loc) & 1u;
                    float q_eff = (float)q4 + (sb ? gamma_f : -gamma_f);
                    float x_tilde = fmaf(d_ptr[j], q_eff, o_ptr[j]);
                    double diff = (double)((float)x_ptr[j]) - (double)x_tilde;
                    err2 += diff * diff;
                }
            }
        }
        __half eta_h = is_valid ? double_to_half_ru_pos(sqrt(err2)) : __ushort_as_half(0x0000u);
        out_ball_meta[(int64_t)m * N_pad + global_row] = __halves2half2(gamma_h, eta_h);
    }
}

__global__ void estimate_query_plan_kernel(
    const TileHeader* __restrict__ headers,     // [N_wtiles]
    const int32_t* __restrict__ cluster_t_off,  // [nlist + 1]
    const int32_t* __restrict__ cand_clusters,  // [B, P_cand]
    const uint32_t* __restrict__ op_flags_arr,
    const int32_t* __restrict__ l0_arr,
    const int32_t* __restrict__ r0_arr,
    const int32_t* __restrict__ l1_arr,
    const int32_t* __restrict__ r1_arr,
    const uint64_t* __restrict__ cat_mask_arr,
    const int32_t* __restrict__ w0_arr,
    const int32_t* __restrict__ w3_arr,
    const int32_t* __restrict__ alg_bound_arr,
    int32_t B,
    int32_t N_wtiles,
    int32_t P_cand,
    int32_t base_nprobe,
    bool enable_prewarm,
    int32_t* __restrict__ out_ordered_probes,   // [B, P_cand]
    int32_t* __restrict__ out_n_alive_probes,   // [B]
    float* __restrict__ out_est_global_sel,     // [B]
    float* __restrict__ out_est_probe_surv,     // [B]
    float* __restrict__ out_cluster_skew        // [B]
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x; // 0..127

    uint32_t op_flags = op_flags_arr[b];
    int32_t l0 = l0_arr[b], r0 = r0_arr[b];
    int32_t l1 = l1_arr[b], r1 = r1_arr[b];
    uint64_t cm = cat_mask_arr[b];
    int32_t w0 = w0_arr[b], w3 = w3_arr[b], ab = alg_bound_arr[b];

    // 1. Sample 1/16th of global tiles to estimate global selectivity in ~3 us
    float loc_hit = 0.0f;
    float loc_tot = 0.0f;
    for (int wt = tid * 16; wt < N_wtiles; wt += blockDim.x * 16) {
        TileHeader th = headers[wt];
        loc_tot += (float)th.valid_cnt;
        int st = check_tile_header(th, op_flags, l0, r0, l1, r1, cm, w0, w3, ab);
        if (st == 2) {
            loc_hit += (float)th.valid_cnt;
        } else if (st == 1) {
            float frac = 1.0f;
            if (op_flags & PRED_COL0_RANGE) {
                float span = (float)((int32_t)th.c0_max - (int32_t)th.c0_min + 1);
                float ov = (float)(min((int32_t)th.c0_max, r0) - max((int32_t)th.c0_min, l0) + 1);
                frac *= fmaxf(0.0f, fminf(1.0f, ov / span));
            }
            if (op_flags & PRED_COL1_RANGE) {
                float span = (float)((int32_t)th.c1_max - (int32_t)th.c1_min + 1);
                float ov = (float)(min((int32_t)th.c1_max, r1) - max((int32_t)th.c1_min, l1) + 1);
                frac *= fmaxf(0.0f, fminf(1.0f, ov / span));
            }
            if (op_flags & PRED_COL2_CAT) {
                int pop_inter = __popcll(th.cat_bloom64 & cm);
                int pop_tot   = max(1, __popcll(th.cat_bloom64));
                frac *= (float)pop_inter / (float)pop_tot;
            }
            if (op_flags & PRED_ALG_EXPR) {
                frac *= 0.5f;
            }
            loc_hit += frac * (float)th.valid_cnt;
        }
    }

    __shared__ float sh_red_hit[128];
    __shared__ float sh_red_tot[128];
    sh_red_hit[tid] = loc_hit;
    sh_red_tot[tid] = loc_tot;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) {
            sh_red_hit[tid] += sh_red_hit[tid + s];
            sh_red_tot[tid] += sh_red_tot[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_est_global_sel[b] = (sh_red_tot[0] > 0.0f) ? (sh_red_hit[0] / sh_red_tot[0]) : 0.0f;
    }

    // 2. Evaluate each candidate cluster in cand_clusters[b, 0..P_cand-1]
    // P_cand <= 512
    __shared__ float sh_csurv[512];
    __shared__ uint8_t sh_calive[512];
    int P_lim = min(P_cand, 512);

    for (int p = tid; p < P_lim; p += blockDim.x) {
        int c = cand_clusters[b * P_cand + p];
        if (c < 0) {
            sh_csurv[p] = 0.0f;
            sh_calive[p] = 0;
            continue;
        }
        int t0 = cluster_t_off[c];
        int t1 = cluster_t_off[c + 1];
        float c_surv = 0.0f;
        uint8_t any_alive = 0;
        for (int wt = t0; wt < t1; ++wt) {
            TileHeader th = headers[wt];
            int st = check_tile_header(th, op_flags, l0, r0, l1, r1, cm, w0, w3, ab);
            if (st == 2) {
                any_alive = 1;
                c_surv += (float)th.valid_cnt;
            } else if (st == 1) {
                any_alive = 1;
                float frac = 1.0f;
                if (op_flags & PRED_COL0_RANGE) {
                    float span = (float)((int32_t)th.c0_max - (int32_t)th.c0_min + 1);
                    float ov = (float)(min((int32_t)th.c0_max, r0) - max((int32_t)th.c0_min, l0) + 1);
                    frac *= fmaxf(0.0f, fminf(1.0f, ov / span));
                }
                if (op_flags & PRED_COL1_RANGE) {
                    float span = (float)((int32_t)th.c1_max - (int32_t)th.c1_min + 1);
                    float ov = (float)(min((int32_t)th.c1_max, r1) - max((int32_t)th.c1_min, l1) + 1);
                    frac *= fmaxf(0.0f, fminf(1.0f, ov / span));
                }
                if (op_flags & PRED_COL2_CAT) {
                    frac *= (float)__popcll(th.cat_bloom64 & cm) / (float)max(1, __popcll(th.cat_bloom64));
                }
                if (op_flags & PRED_ALG_EXPR) frac *= 0.5f;
                c_surv += frac * (float)th.valid_cnt;
            }
        }
        sh_csurv[p] = c_surv;
        sh_calive[p] = any_alive;
    }
    __syncthreads();

    if (tid == 0) {
        // Compute survivor stats on base_nprobe prefix + filter out 100%-dead clusters losslessly
        int n_base = min(base_nprobe, P_lim);
        float sum_s = 0.0f;
        float sum_sq = 0.0f;
        for (int p = 0; p < n_base; ++p) {
            float v = sh_csurv[p];
            sum_s += v;
            sum_sq += v * v;
        }
        float mean_s = (n_base > 0) ? (sum_s / (float)n_base) : 0.0f;
        float var_s  = (n_base > 0) ? fmaxf(0.0f, sum_sq / (float)n_base - mean_s * mean_s) : 0.0f;
        out_est_probe_surv[b] = sum_s;
        out_cluster_skew[b]   = sqrtf(var_s) / (mean_s + 1.0f);

        // Adaptive Equal-Recall Probe Expansion: if base_nprobe has too few expected survivors (< 48),
        // expand up to P_lim until cumulative expected survivors reach target or P_lim exhausted!
        int target_probes = n_base;
        float cum_s = sum_s;
        while (target_probes < P_lim && cum_s < 64.0f) {
            cum_s += sh_csurv[target_probes];
            target_probes++;
        }

        // Compact alive clusters within [0 .. target_probes - 1], optionally promoting 2 rich clusters first
        int out_cnt = 0;
        int pre1 = -1, pre2 = -1;
        if (enable_prewarm) {
            float thresh = fmaxf(2.0f, 0.3f * mean_s);
            for (int p = 0; p < target_probes; ++p) {
                if (sh_calive[p] && sh_csurv[p] >= thresh) {
                    if (pre1 < 0) pre1 = p;
                    else if (pre2 < 0) { pre2 = p; break; }
                }
            }
        }
        int32_t* dst = out_ordered_probes + b * P_cand;
        if (pre1 >= 0) dst[out_cnt++] = cand_clusters[b * P_cand + pre1];
        if (pre2 >= 0) dst[out_cnt++] = cand_clusters[b * P_cand + pre2];
        for (int p = 0; p < target_probes; ++p) {
            if (p == pre1 || p == pre2) continue;
            if (sh_calive[p]) {
                dst[out_cnt++] = cand_clusters[b * P_cand + p];
            }
        }
        for (int p = out_cnt; p < P_cand; ++p) dst[p] = -1;
        out_n_alive_probes[b] = out_cnt;
    }
}

template <typename VecT>
__device__ __forceinline__ void refine_and_update_warp(
    uint32_t surv_mask,
    uint32_t cand_row,
    int lane,
    int D,
    int k,
    const float* __restrict__ sh_q,
    const VecT* __restrict__ exact_vecs,
    const int64_t* __restrict__ perm_to_orig_id,
    volatile float* sh_tau,
    int* sh_lock,
    float* sh_topk_d,
    int64_t* sh_topk_i,
    int32_t* g_tau_int,
    uint64_t* g_telemetry,
    bool rec_tel
) {
    if (rec_tel && lane == 0 && surv_mask != 0u) {
        atomicAdd((unsigned long long*)&g_telemetry[10], (unsigned long long)__popc(surv_mask));
    }
    long long t_ref0 = (rec_tel && lane == 0) ? clock64() : 0LL;
    while (surv_mask != 0u) {
        int who = __ffs(surv_mask) - 1;
        surv_mask &= (surv_mask - 1u);
        uint32_t row_id = __shfl_sync(0xffffffffu, cand_row, who);

        // Quick pre-check against latest shared tau before touching PCIe / DRAM
        float cur_tau = *sh_tau;
        const VecT* xv = exact_vecs + (int64_t)row_id * D;
        float part_d2 = 0.0f;
        for (int j = lane; j < D; j += 32) {
            float diff = sh_q[j] - (float)xv[j];
            part_d2 = fmaf(diff, diff, part_d2);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part_d2 += __shfl_down_sync(0xffffffffu, part_d2, off);
        }

        if (lane == 0) {
            if (part_d2 < *sh_tau || sh_topk_i[k - 1] < 0) {
                int64_t orig_id = perm_to_orig_id[row_id];
                while (atomicCAS(sh_lock, 0, 1) != 0) {}
                if (part_d2 < sh_topk_d[k - 1] || sh_topk_i[k - 1] < 0) {
                    // Check duplicate ID safety (should never happen as tiles are disjoint)
                    int pos = k - 1;
                    while (pos > 0 && (sh_topk_i[pos - 1] < 0 || part_d2 < sh_topk_d[pos - 1])) {
                        sh_topk_d[pos] = sh_topk_d[pos - 1];
                        sh_topk_i[pos] = sh_topk_i[pos - 1];
                        pos--;
                    }
                    sh_topk_d[pos] = part_d2;
                    sh_topk_i[pos] = orig_id;
                    if (sh_topk_i[k - 1] >= 0) {
                        float new_kth = sh_topk_d[k - 1];
                        *sh_tau = new_kth;
                        __threadfence_block();
                        if (new_kth >= 0.0f && isfinite(new_kth)) {
                            atomicMin(g_tau_int, __float_as_int(new_kth));
                        }
                    }
                }
                __threadfence_block();
                atomicExch(sh_lock, 0);
            }
        }
        __syncwarp();
    }
    if (rec_tel && lane == 0) {
        atomicAdd((unsigned long long*)&g_telemetry[13], (unsigned long long)(clock64() - t_ref0));
    }
}

__device__ __forceinline__ uint32_t eval_subspace_balls_warp(
    uint32_t active_mask,
    uint32_t cand_row,
    int32_t cand_cid,
    int lane,
    int D,
    int D_pad,
    int M,
    int64_t N_pad,
    const int32_t* __restrict__ block_bounds,
    const float* __restrict__ sh_q,
    const float* __restrict__ cluster_o,
    const float* __restrict__ cluster_delta,
    const uint4* __restrict__ int4_codes,
    const uint32_t* __restrict__ sign_bits,
    const __half2* __restrict__ ball_meta,
    float cur_tau,
    uint64_t* g_telemetry,
    bool rec_tel,
    int start_block = 0,
    float init_lb2 = 0.0f
) {
    bool alive = ((active_mask >> lane) & 1u) != 0u;
    if (__ballot_sync(0xffffffffu, alive) == 0u) return 0u;
    long long t_sk0 = (rec_tel && lane == 0) ? clock64() : 0LL;

    bool has_tau = isfinite(cur_tau);
    if (!has_tau && !rec_tel) return active_mask;
    float tau_safe = has_tau ? (cur_tau * (1.0f + 3.5e-5f) + 1e-6f) : FLT_MAX;
    float lb2_acc = init_lb2;
    const float* my_o = cluster_o + (int64_t)cand_cid * D_pad;
    const float* my_d = cluster_delta + (int64_t)cand_cid * D_pad;

    for (int m = start_block; m < M; ++m) {
        int u_beg = block_bounds[m];
        int u_end = block_bounds[m + 1];
        int blk_dims = (u_end - u_beg) * 32;
        float eps_rel = (float)(blk_dims + 8) * 5.9604645e-8f;

        if (alive) {
            __half2 meta = ball_meta[(int64_t)m * N_pad + cand_row];
            float gamma_f = __half2float(__low2half(meta));
            float eta_f   = __half2float(__high2half(meta));

            float dist2_tilde = 0.0f;
            for (int u = u_beg; u < u_end; ++u) {
                uint4 pack4 = int4_codes[(int64_t)u * N_pad + cand_row];
                uint32_t s_word = sign_bits[(int64_t)u * N_pad + cand_row];
                uint32_t words[4] = {pack4.x, pack4.y, pack4.z, pack4.w};

                #pragma unroll
                for (int d_loc = 0; d_loc < 32; ++d_loc) {
                    int j = u * 32 + d_loc;
                    if (j < D) {
                        uint32_t q4 = (words[d_loc >> 3] >> ((d_loc & 7) * 4)) & 0xFu;
                        uint32_t sb = (s_word >> d_loc) & 1u;
                        float q_eff = (float)q4 + (sb ? gamma_f : -gamma_f);
                        float x_tilde = fmaf(__ldg(&my_d[j]), q_eff, __ldg(&my_o[j]));
                        float diff = sh_q[j] - x_tilde;
                        dist2_tilde = fmaf(diff, diff, dist2_tilde);
                    }
                }
            }
            float dist_tilde = sqrtf(dist2_tilde);
            float lb_m = fmaxf(0.0f, (1.0f - eps_rel) * dist_tilde - eta_f);
            lb2_acc = fmaf(lb_m, lb_m, lb2_acc);
            if (has_tau && lb2_acc > tau_safe) {
                alive = false;
            }
        }

        uint32_t cur_alive_mask = __ballot_sync(0xffffffffu, alive);
        if (rec_tel && lane == 0 && m < 4) {
            atomicAdd((unsigned long long*)&g_telemetry[6 + m], (unsigned long long)__popc(cur_alive_mask));
        }
        if (cur_alive_mask == 0u) {
            if (rec_tel && lane == 0) {
                atomicAdd((unsigned long long*)&g_telemetry[12], (unsigned long long)(clock64() - t_sk0));
            }
            return 0u;
        }
    }
    if (rec_tel && lane == 0) {
        atomicAdd((unsigned long long*)&g_telemetry[12], (unsigned long long)(clock64() - t_sk0));
    }
    return __ballot_sync(0xffffffffu, alive);
}

template <typename VecT>
__global__ void prism_search_kernel(
    const float* __restrict__ queries,           // [B, D]
    const TileHeader* __restrict__ headers,      // [N_wtiles]
    const uint64_t* __restrict__ slap_cols,      // [N_pad]
    const uint4* __restrict__ int4_codes,        // [chunks32, N_pad]
    const uint32_t* __restrict__ sign_bits,      // [chunks32, N_pad]
    const __half2* __restrict__ ball_meta,       // [M, N_pad]
    const float* __restrict__ cluster_o,         // [nlist, D_pad]
    const float* __restrict__ cluster_delta,     // [nlist, D_pad]
    const int32_t* __restrict__ cluster_t_off,   // [nlist + 1]
    const int32_t* __restrict__ block_bounds_g,  // [M + 1]
    const VecT* __restrict__ exact_vecs,         // [N_pad, D] (Device or Host Pinned UVA)
    const int64_t* __restrict__ perm_to_orig_id, // [N_pad]
    const int32_t* __restrict__ probe_lists,     // [B, P_stride] (ignored if regime == 1)
    const int32_t* __restrict__ n_probes_arr,    // [B]
    const uint32_t* __restrict__ op_flags_arr,
    const int32_t* __restrict__ l0_arr,
    const int32_t* __restrict__ r0_arr,
    const int32_t* __restrict__ l1_arr,
    const int32_t* __restrict__ r1_arr,
    const uint64_t* __restrict__ cat_mask_arr,
    const int32_t* __restrict__ w0_arr,
    const int32_t* __restrict__ w3_arr,
    const int32_t* __restrict__ alg_bound_arr,
    int32_t B,
    int64_t N_pad,
    int32_t nlist,
    int32_t D,
    int32_t D_pad,
    int32_t M,
    int32_t k,
    int32_t P_stride,
    int32_t G_split,
    int32_t regime,                 // 1 = Global Pushdown, 2 = Warp-Compacted IVF, 3 = Sketch-First IVF
    int32_t sketch_mode,            // 0 = Exact PreFilter (no sketch), >=1 = Subspace Ball Sketch
    bool enable_warp_compaction,
    bool enable_tile_header,
    bool rec_tel,
    int32_t* __restrict__ g_tau_int,             // [B] initialized to 0x7f7fffff (FLT_MAX)
    float* __restrict__ partial_d,               // [B, G_split, k]
    int64_t* __restrict__ partial_i,             // [B, G_split, k]
    uint64_t* __restrict__ g_telemetry           // [16]
) {
    int split_id = blockIdx.x;
    int b        = blockIdx.y;
    if (b >= B || split_id >= G_split) return;

    int tid  = threadIdx.x;        // 0..255
    int wid  = tid >> 5;           // 0..7
    int lane = tid & 31;           // 0..31

    __shared__ float sh_q[1024];
    __shared__ uint64_t sh_ring[8][64];
    __shared__ float sh_topk_d[128];
    __shared__ int64_t sh_topk_i[128];
    __shared__ int32_t sh_block_bounds[33];
    __shared__ volatile float sh_tau;
    __shared__ int sh_lock;

    for (int j = tid; j < D_pad; j += blockDim.x) {
        sh_q[j] = (j < D) ? queries[(int64_t)b * D + j] : 0.0f;
    }
    if (tid < k) {
        sh_topk_d[tid] = FLT_MAX;
        sh_topk_i[tid] = -1LL;
    }
    if (tid <= M && tid <= 32) {
        sh_block_bounds[tid] = block_bounds_g[tid];
    }
    if (tid == 0) {
        sh_tau = FLT_MAX;
        sh_lock = 0;
    }
    __syncthreads();

    uint32_t op_flags = op_flags_arr[b];
    int32_t l0 = l0_arr[b], r0 = r0_arr[b];
    int32_t l1 = l1_arr[b], r1 = r1_arr[b];
    uint64_t cm = cat_mask_arr[b];
    int32_t w0 = w0_arr[b], w3 = w3_arr[b], ab = alg_bound_arr[b];

    int total_clusters = (regime == 1) ? nlist : n_probes_arr[b];
    int ring_cnt = 0;

    for (int idx_c = split_id; idx_c < total_clusters; idx_c += G_split) {
        int c = (regime == 1) ? idx_c : probe_lists[(int64_t)b * P_stride + idx_c];
        if (c < 0 || c >= nlist) continue;

        int t_beg = cluster_t_off[c];
        int t_end = cluster_t_off[c + 1];
        if (t_beg >= t_end) continue;

        // In Regime 1 (Global Scan), quickly check if ANY tile in cluster c survives
        if (regime == 1 && enable_tile_header) {
            __shared__ int sh_cluster_alive;
            if (tid == 0) sh_cluster_alive = 0;
            __syncthreads();
            for (int wt = t_beg + tid; wt < t_end; wt += blockDim.x) {
                if (check_tile_header(headers[wt], op_flags, l0, r0, l1, r1, cm, w0, w3, ab) > 0) {
                    atomicExch(&sh_cluster_alive, 1);
                    break;
                }
            }
            __syncthreads();
            if (sh_cluster_alive == 0) {
                if (rec_tel && tid == 0) {
                    uint64_t nt = (uint64_t)(t_end - t_beg);
                    atomicAdd((unsigned long long*)&g_telemetry[0], nt);
                    atomicAdd((unsigned long long*)&g_telemetry[1], nt);
                }
                continue;
            }
        }

        // Sync cross-CTA global threshold into sh_tau
        if (tid == 0) {
            float g_t = __int_as_float(g_tau_int[b]);
            if (g_t < sh_tau) sh_tau = g_t;
        }
        __syncthreads();

        const float* co_c = cluster_o + (int64_t)c * D_pad;
        const float* cd_c = cluster_delta + (int64_t)c * D_pad;

        for (int wt = t_beg + wid; wt < t_end; wt += 8) {
            long long t_flt0 = (rec_tel && lane == 0) ? clock64() : 0LL;
            TileHeader th = headers[wt];
            int vcnt = (int)th.valid_cnt;
            if (vcnt <= 0) continue;
            uint32_t valid_lane_mask = (vcnt >= 32) ? 0xffffffffu : ((1u << vcnt) - 1u);
            uint32_t row_id = th.row_offset + lane;

            if (rec_tel && lane == 0) {
                atomicAdd((unsigned long long*)&g_telemetry[0], 1ULL);
            }

            int st = ((th.flags & 1u) != 0u) ? 1 : 1; // BOUNDARY if tile header disabled
            if (enable_tile_header) {
                if (lane == 0) {
                    st = check_tile_header(th, op_flags, l0, r0, l1, r1, cm, w0, w3, ab);
                }
                st = __shfl_sync(0xffffffffu, st, 0);
                if (st == 0) {
                    if (rec_tel && lane == 0) {
                        atomicAdd((unsigned long long*)&g_telemetry[1], 1ULL);
                        atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_flt0));
                    }
                    continue;
                }
                if (st == 2 && rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[2], 1ULL);
                }
            }

            // REGIME 3: Sketch-First Late-Predicate Pipeline (for High Selectivity)
            if (regime == 3 && sketch_mode >= 1) {
                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_flt0));
                    atomicAdd((unsigned long long*)&g_telemetry[3], 1ULL);
                    atomicAdd((unsigned long long*)&g_telemetry[4], (unsigned long long)vcnt);
                    atomicAdd((unsigned long long*)&g_telemetry[5], 1ULL);
                }
                long long t_b0 = (rec_tel && lane == 0) ? clock64() : 0LL;
                float cur_tau = fminf(sh_tau, __int_as_float(g_tau_int[b]));
                bool has_tau = isfinite(cur_tau);
                float tau_safe = has_tau ? (cur_tau * (1.0f + 3.5e-5f) + 1e-6f) : FLT_MAX;

                bool alive0 = (lane < vcnt);
                float lb2_0 = 0.0f;
                int u0_beg = sh_block_bounds[0];
                int u0_end = sh_block_bounds[1];
                float eps0 = (float)((u0_end - u0_beg) * 32 + 8) * 5.9604645e-8f;

                if (alive0 && (has_tau || rec_tel)) {
                    __half2 meta0 = ball_meta[row_id];
                    float gamma0 = __half2float(__low2half(meta0));
                    float eta0   = __half2float(__high2half(meta0));
                    float d2_t   = 0.0f;
                    for (int u = u0_beg; u < u0_end; ++u) {
                        uint4 p4 = int4_codes[(int64_t)u * N_pad + row_id];
                        uint32_t sw = sign_bits[(int64_t)u * N_pad + row_id];
                        uint32_t words[4] = {p4.x, p4.y, p4.z, p4.w};
                        #pragma unroll
                        for (int d_loc = 0; d_loc < 32; ++d_loc) {
                            int j = u * 32 + d_loc;
                            if (j < D) {
                                uint32_t q4 = (words[d_loc >> 3] >> ((d_loc & 7) * 4)) & 0xFu;
                                uint32_t sb = (sw >> d_loc) & 1u;
                                float q_eff = (float)q4 + (sb ? gamma0 : -gamma0);
                                float xt = fmaf(__ldg(&cd_c[j]), q_eff, __ldg(&co_c[j]));
                                float df = sh_q[j] - xt;
                                d2_t = fmaf(df, df, d2_t);
                            }
                        }
                    }
                    float lb0 = fmaxf(0.0f, (1.0f - eps0) * sqrtf(d2_t) - eta0);
                    lb2_0 = lb0 * lb0;
                    if (has_tau && lb2_0 > tau_safe) alive0 = false;
                }
                uint32_t m_after_b0 = __ballot_sync(0xffffffffu, alive0);
                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[6], (unsigned long long)__popc(m_after_b0));
                    atomicAdd((unsigned long long*)&g_telemetry[12], (unsigned long long)(clock64() - t_b0));
                }
                if (m_after_b0 == 0u) continue;

                // Evaluate SQL predicate ONLY on Block-0 survivors!
                long long t_lp0 = (rec_tel && lane == 0) ? clock64() : 0LL;
                if (st == 1 && alive0) {
                    uint64_t packed = slap_cols[row_id];
                    alive0 = eval_slap_tuple(packed, op_flags, l0, r0, l1, r1, cm, w0, w3, ab);
                }
                uint32_t m_pred_b0 = __ballot_sync(0xffffffffu, alive0);
                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_lp0));
                }
                if (m_pred_b0 == 0u) continue;

                uint32_t surv_m = (M > 1)
                    ? eval_subspace_balls_warp(
                        m_pred_b0, row_id, c, lane, D, D_pad, M, N_pad, sh_block_bounds,
                        sh_q, cluster_o, cluster_delta, int4_codes, sign_bits, ball_meta,
                        cur_tau, g_telemetry, rec_tel, 1, lb2_0
                    )
                    : m_pred_b0;
                if (surv_m != 0u) {
                    refine_and_update_warp(
                        surv_m, row_id, lane, D, k, sh_q, exact_vecs, perm_to_orig_id,
                        &sh_tau, &sh_lock, sh_topk_d, sh_topk_i, &g_tau_int[b], g_telemetry, rec_tel
                    );
                }
                continue;
            }

            // REGIME 1 & REGIME 2: Predicate Pushdown + Cross-Cluster Persistent Warp Compaction
            uint32_t pred_mask = 0u;
            if (st == 2) {
                pred_mask = valid_lane_mask;
            } else {
                bool ok = false;
                if (lane < vcnt) {
                    uint64_t packed = slap_cols[row_id];
                    ok = eval_slap_tuple(packed, op_flags, l0, r0, l1, r1, cm, w0, w3, ab);
                }
                pred_mask = __ballot_sync(0xffffffffu, ok);
            }
            if (pred_mask == 0u) {
                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_flt0));
                }
                continue;
            }

            int n_pass = __popc(pred_mask);
            if (rec_tel && lane == 0) {
                atomicAdd((unsigned long long*)&g_telemetry[3], 1ULL);
                atomicAdd((unsigned long long*)&g_telemetry[4], (unsigned long long)n_pass);
            }

            if (sketch_mode == 0) {
                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_flt0));
                }
                refine_and_update_warp(
                    pred_mask, row_id, lane, D, k, sh_q, exact_vecs, perm_to_orig_id,
                    &sh_tau, &sh_lock, sh_topk_d, sh_topk_i, &g_tau_int[b], g_telemetry, rec_tel
                );
                continue;
            }

            if (!enable_warp_compaction) {
                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[5], 1ULL);
                    atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_flt0));
                }
                float cur_tau = fminf(sh_tau, __int_as_float(g_tau_int[b]));
                uint32_t surv_m = eval_subspace_balls_warp(
                    pred_mask, row_id, c, lane, D, D_pad, M, N_pad, sh_block_bounds,
                    sh_q, cluster_o, cluster_delta, int4_codes, sign_bits, ball_meta,
                    cur_tau, g_telemetry, rec_tel, 0, 0.0f
                );
                if (surv_m != 0u) {
                    refine_and_update_warp(
                        surv_m, row_id, lane, D, k, sh_q, exact_vecs, perm_to_orig_id,
                        &sh_tau, &sh_lock, sh_topk_d, sh_topk_i, &g_tau_int[b], g_telemetry, rec_tel
                    );
                }
                continue;
            }

            // Cross-Cluster Persistent Per-Warp Survivor Stack (`sh_ring[wid][64]`)
            if (((pred_mask >> lane) & 1u) != 0u) {
                int prefix = __popc(pred_mask & ((1u << lane) - 1u));
                sh_ring[wid][ring_cnt + prefix] = ((uint64_t)(uint32_t)c << 32) | (uint64_t)row_id;
            }
            ring_cnt += n_pass;
            __syncwarp();
            if (rec_tel && lane == 0) {
                atomicAdd((unsigned long long*)&g_telemetry[11], (unsigned long long)(clock64() - t_flt0));
            }

            if (ring_cnt >= 32) {
                ring_cnt -= 32;
                uint64_t packed_cand = sh_ring[wid][ring_cnt + lane];
                uint32_t comp_row = (uint32_t)(packed_cand & 0xffffffffULL);
                int32_t  comp_cid = (int32_t)(packed_cand >> 32);
                __syncwarp();

                if (rec_tel && lane == 0) {
                    atomicAdd((unsigned long long*)&g_telemetry[5], 1ULL);
                }
                float cur_tau = fminf(sh_tau, __int_as_float(g_tau_int[b]));
                uint32_t surv_m = eval_subspace_balls_warp(
                    0xffffffffu, comp_row, comp_cid, lane, D, D_pad, M, N_pad, sh_block_bounds,
                    sh_q, cluster_o, cluster_delta, int4_codes, sign_bits, ball_meta,
                    cur_tau, g_telemetry, rec_tel, 0, 0.0f
                );
                if (surv_m != 0u) {
                    refine_and_update_warp(
                        surv_m, comp_row, lane, D, k, sh_q, exact_vecs, perm_to_orig_id,
                        &sh_tau, &sh_lock, sh_topk_d, sh_topk_i, &g_tau_int[b], g_telemetry, rec_tel
                    );
                }
            }
        }

        // Pass-1 Threshold Bootstrap: if tau is still +Inf after prewarm cluster and we have >= k survivors,
        // flush once so tau becomes finite immediately, then keep persistent full-warp compaction for all rest!
        if (sketch_mode >= 1 && enable_warp_compaction && !isfinite(sh_tau) && ring_cnt >= k) {
            uint32_t tail_mask = (1u << ring_cnt) - 1u;
            uint64_t packed_cand = (lane < ring_cnt) ? sh_ring[wid][lane] : 0ULL;
            uint32_t comp_row = (uint32_t)(packed_cand & 0xffffffffULL);
            int32_t  comp_cid = (int32_t)(packed_cand >> 32);
            ring_cnt = 0;
            __syncwarp();

            if (rec_tel && lane == 0) {
                atomicAdd((unsigned long long*)&g_telemetry[5], 1ULL);
            }
            float cur_tau = fminf(sh_tau, __int_as_float(g_tau_int[b]));
            uint32_t surv_m = eval_subspace_balls_warp(
                tail_mask, comp_row, comp_cid, lane, D, D_pad, M, N_pad, sh_block_bounds,
                sh_q, cluster_o, cluster_delta, int4_codes, sign_bits, ball_meta,
                cur_tau, g_telemetry, rec_tel, 0, 0.0f
            );
            if (surv_m != 0u) {
                refine_and_update_warp(
                    surv_m, comp_row, lane, D, k, sh_q, exact_vecs, perm_to_orig_id,
                    &sh_tau, &sh_lock, sh_topk_d, sh_topk_i, &g_tau_int[b], g_telemetry, rec_tel
                );
            }
        }
        __syncthreads();
    }

    // Final single tail drain at end of entire query (<= 31 leftover survivors across all clusters!)
    if (sketch_mode >= 1 && enable_warp_compaction && ring_cnt > 0) {
        uint32_t tail_mask = (1u << ring_cnt) - 1u;
        uint64_t packed_cand = (lane < ring_cnt) ? sh_ring[wid][lane] : 0ULL;
        uint32_t comp_row = (uint32_t)(packed_cand & 0xffffffffULL);
        int32_t  comp_cid = (int32_t)(packed_cand >> 32);
        ring_cnt = 0;
        __syncwarp();

        if (rec_tel && lane == 0) {
            atomicAdd((unsigned long long*)&g_telemetry[5], 1ULL);
        }
        float cur_tau = fminf(sh_tau, __int_as_float(g_tau_int[b]));
        uint32_t surv_m = eval_subspace_balls_warp(
            tail_mask, comp_row, comp_cid, lane, D, D_pad, M, N_pad, sh_block_bounds,
            sh_q, cluster_o, cluster_delta, int4_codes, sign_bits, ball_meta,
            cur_tau, g_telemetry, rec_tel, 0, 0.0f
        );
        if (surv_m != 0u) {
            refine_and_update_warp(
                surv_m, comp_row, lane, D, k, sh_q, exact_vecs, perm_to_orig_id,
                &sh_tau, &sh_lock, sh_topk_d, sh_topk_i, &g_tau_int[b], g_telemetry, rec_tel
            );
        }
    }
    __syncthreads();

    // Write out CTA partial top-k
    int64_t out_base = ((int64_t)b * G_split + split_id) * k;
    for (int slot = tid; slot < k; slot += blockDim.x) {
        partial_d[out_base + slot] = sh_topk_d[slot];
        partial_i[out_base + slot] = sh_topk_i[slot];
    }
}

__global__ void merge_partial_topk_kernel(
    const float* __restrict__ partial_d,    // [B, G_split, k]
    const int64_t* __restrict__ partial_i,  // [B, G_split, k]
    int32_t B,
    int32_t G_split,
    int32_t k,
    float* __restrict__ out_d,              // [B, k]
    int64_t* __restrict__ out_i             // [B, k]
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    float best_d[128];
    int64_t best_i[128];
    for (int s = 0; s < k; ++s) {
        best_d[s] = FLT_MAX;
        best_i[s] = -1LL;
    }

    const float* pd = partial_d + (int64_t)b * G_split * k;
    const int64_t* pi = partial_i + (int64_t)b * G_split * k;

    for (int g = 0; g < G_split; ++g) {
        for (int s = 0; s < k; ++s) {
            int64_t cid = pi[g * k + s];
            float   cd  = pd[g * k + s];
            if (cid < 0) break;
            if (cd < best_d[k - 1] || best_i[k - 1] < 0) {
                int pos = k - 1;
                while (pos > 0 && (best_i[pos - 1] < 0 || cd < best_d[pos - 1])) {
                    best_d[pos] = best_d[pos - 1];
                    best_i[pos] = best_i[pos - 1];
                    pos--;
                }
                best_d[pos] = cd;
                best_i[pos] = cid;
            }
        }
    }

    for (int s = 0; s < k; ++s) {
        out_d[(int64_t)b * k + s] = best_d[s];
        out_i[(int64_t)b * k + s] = best_i[s];
    }
}

void encode_prism_tiles(
    torch::Tensor vecs_chunk,
    torch::Tensor c0_chunk,
    torch::Tensor c1_chunk,
    torch::Tensor c2_chunk,
    torch::Tensor c3_chunk,
    torch::Tensor tile_valid,
    torch::Tensor tile_clusters,
    torch::Tensor cluster_o,
    torch::Tensor cluster_delta,
    torch::Tensor block_bounds,
    int64_t base_tile_id,
    int64_t N_pad,
    int64_t D,
    int64_t D_pad,
    int64_t M,
    bool use_sign_bit,
    torch::Tensor out_headers_u8,
    torch::Tensor out_slap_cols_i64,
    torch::Tensor out_int4_i32,
    torch::Tensor out_sign_i32,
    torch::Tensor out_ball_meta_i32
) {
    int32_t n_tiles = (int32_t)tile_valid.size(0);
    if (n_tiles <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto* hdr_ptr  = reinterpret_cast<TileHeader*>(out_headers_u8.data_ptr<uint8_t>());
    auto* slap_ptr = reinterpret_cast<uint64_t*>(out_slap_cols_i64.data_ptr<int64_t>());
    auto* i4_ptr   = reinterpret_cast<uint4*>(out_int4_i32.data_ptr<int32_t>());
    auto* sg_ptr   = reinterpret_cast<uint32_t*>(out_sign_i32.data_ptr<int32_t>());
    auto* bm_ptr   = reinterpret_cast<__half2*>(out_ball_meta_i32.data_ptr<int32_t>());

    if (vecs_chunk.scalar_type() == torch::kFloat32) {
        encode_prism_tiles_kernel<float><<<n_tiles, 32, 0, stream>>>(
            vecs_chunk.data_ptr<float>(),
            static_cast<const uint16_t*>(c0_chunk.data_ptr()),
            static_cast<const uint16_t*>(c1_chunk.data_ptr()),
            static_cast<const uint16_t*>(c2_chunk.data_ptr()),
            static_cast<const uint16_t*>(c3_chunk.data_ptr()),
            static_cast<const uint16_t*>(tile_valid.data_ptr()),
            tile_clusters.data_ptr<int32_t>(),
            nullptr,
            cluster_o.data_ptr<float>(),
            cluster_delta.data_ptr<float>(),
            block_bounds.data_ptr<int32_t>(),
            (int32_t)base_tile_id, n_tiles, N_pad, (int32_t)D, (int32_t)D_pad, (int32_t)M,
            use_sign_bit, hdr_ptr, slap_ptr, i4_ptr, sg_ptr, bm_ptr
        );
    } else if (vecs_chunk.scalar_type() == torch::kUInt8) {
        encode_prism_tiles_kernel<uint8_t><<<n_tiles, 32, 0, stream>>>(
            vecs_chunk.data_ptr<uint8_t>(),
            static_cast<const uint16_t*>(c0_chunk.data_ptr()),
            static_cast<const uint16_t*>(c1_chunk.data_ptr()),
            static_cast<const uint16_t*>(c2_chunk.data_ptr()),
            static_cast<const uint16_t*>(c3_chunk.data_ptr()),
            static_cast<const uint16_t*>(tile_valid.data_ptr()),
            tile_clusters.data_ptr<int32_t>(),
            nullptr,
            cluster_o.data_ptr<float>(),
            cluster_delta.data_ptr<float>(),
            block_bounds.data_ptr<int32_t>(),
            (int32_t)base_tile_id, n_tiles, N_pad, (int32_t)D, (int32_t)D_pad, (int32_t)M,
            use_sign_bit, hdr_ptr, slap_ptr, i4_ptr, sg_ptr, bm_ptr
        );
    } else {
        TORCH_CHECK(false, "Unsupported vecs_chunk dtype");
    }
}

void encode_prism_sparse_tiles(
    torch::Tensor vecs_chunk,
    torch::Tensor c0_chunk,
    torch::Tensor c1_chunk,
    torch::Tensor c2_chunk,
    torch::Tensor c3_chunk,
    torch::Tensor tile_valid,
    torch::Tensor tile_clusters,
    torch::Tensor target_tiles_i32,
    torch::Tensor cluster_o,
    torch::Tensor cluster_delta,
    torch::Tensor block_bounds,
    int64_t N_pad,
    int64_t D,
    int64_t D_pad,
    int64_t M,
    bool use_sign_bit,
    torch::Tensor out_headers_u8,
    torch::Tensor out_slap_cols_i64,
    torch::Tensor out_int4_i32,
    torch::Tensor out_sign_i32,
    torch::Tensor out_ball_meta_i32
) {
    int32_t n_tiles = (int32_t)tile_valid.size(0);
    if (n_tiles <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto* hdr_ptr  = reinterpret_cast<TileHeader*>(out_headers_u8.data_ptr<uint8_t>());
    auto* slap_ptr = reinterpret_cast<uint64_t*>(out_slap_cols_i64.data_ptr<int64_t>());
    auto* i4_ptr   = reinterpret_cast<uint4*>(out_int4_i32.data_ptr<int32_t>());
    auto* sg_ptr   = reinterpret_cast<uint32_t*>(out_sign_i32.data_ptr<int32_t>());
    auto* bm_ptr   = reinterpret_cast<__half2*>(out_ball_meta_i32.data_ptr<int32_t>());

    if (vecs_chunk.scalar_type() == torch::kFloat32) {
        encode_prism_tiles_kernel<float><<<n_tiles, 32, 0, stream>>>(
            vecs_chunk.data_ptr<float>(),
            static_cast<const uint16_t*>(c0_chunk.data_ptr()),
            static_cast<const uint16_t*>(c1_chunk.data_ptr()),
            static_cast<const uint16_t*>(c2_chunk.data_ptr()),
            static_cast<const uint16_t*>(c3_chunk.data_ptr()),
            static_cast<const uint16_t*>(tile_valid.data_ptr()),
            tile_clusters.data_ptr<int32_t>(),
            target_tiles_i32.data_ptr<int32_t>(),
            cluster_o.data_ptr<float>(),
            cluster_delta.data_ptr<float>(),
            block_bounds.data_ptr<int32_t>(),
            0, n_tiles, N_pad, (int32_t)D, (int32_t)D_pad, (int32_t)M,
            use_sign_bit, hdr_ptr, slap_ptr, i4_ptr, sg_ptr, bm_ptr
        );
    } else if (vecs_chunk.scalar_type() == torch::kUInt8) {
        encode_prism_tiles_kernel<uint8_t><<<n_tiles, 32, 0, stream>>>(
            vecs_chunk.data_ptr<uint8_t>(),
            static_cast<const uint16_t*>(c0_chunk.data_ptr()),
            static_cast<const uint16_t*>(c1_chunk.data_ptr()),
            static_cast<const uint16_t*>(c2_chunk.data_ptr()),
            static_cast<const uint16_t*>(c3_chunk.data_ptr()),
            static_cast<const uint16_t*>(tile_valid.data_ptr()),
            tile_clusters.data_ptr<int32_t>(),
            target_tiles_i32.data_ptr<int32_t>(),
            cluster_o.data_ptr<float>(),
            cluster_delta.data_ptr<float>(),
            block_bounds.data_ptr<int32_t>(),
            0, n_tiles, N_pad, (int32_t)D, (int32_t)D_pad, (int32_t)M,
            use_sign_bit, hdr_ptr, slap_ptr, i4_ptr, sg_ptr, bm_ptr
        );
    } else {
        TORCH_CHECK(false, "Unsupported vecs_chunk dtype");
    }
}

void estimate_query_plan(
    torch::Tensor headers_u8,
    torch::Tensor cluster_t_off,
    torch::Tensor cand_clusters,
    torch::Tensor op_flags_i32,
    torch::Tensor l0, torch::Tensor r0,
    torch::Tensor l1, torch::Tensor r1,
    torch::Tensor cat_mask_i64,
    torch::Tensor w0, torch::Tensor w3, torch::Tensor alg_bound,
    int64_t base_nprobe,
    bool enable_prewarm,
    torch::Tensor out_ordered_probes,
    torch::Tensor out_n_alive_probes,
    torch::Tensor out_est_global_sel,
    torch::Tensor out_est_probe_surv,
    torch::Tensor out_cluster_skew
) {
    int32_t B = (int32_t)cand_clusters.size(0);
    int32_t P_cand = (int32_t)cand_clusters.size(1);
    int32_t N_wtiles = (int32_t)headers_u8.size(0);
    if (B <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    estimate_query_plan_kernel<<<B, 128, 0, stream>>>(
        reinterpret_cast<const TileHeader*>(headers_u8.data_ptr<uint8_t>()),
        cluster_t_off.data_ptr<int32_t>(),
        cand_clusters.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(op_flags_i32.data_ptr<int32_t>()),
        l0.data_ptr<int32_t>(), r0.data_ptr<int32_t>(),
        l1.data_ptr<int32_t>(), r1.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(cat_mask_i64.data_ptr<int64_t>()),
        w0.data_ptr<int32_t>(), w3.data_ptr<int32_t>(), alg_bound.data_ptr<int32_t>(),
        B, N_wtiles, P_cand, (int32_t)base_nprobe, enable_prewarm,
        out_ordered_probes.data_ptr<int32_t>(),
        out_n_alive_probes.data_ptr<int32_t>(),
        out_est_global_sel.data_ptr<float>(),
        out_est_probe_surv.data_ptr<float>(),
        out_cluster_skew.data_ptr<float>()
    );
}

void launch_prism_search(
    torch::Tensor queries,
    torch::Tensor headers_u8,
    torch::Tensor slap_cols_i64,
    torch::Tensor int4_i32,
    torch::Tensor sign_i32,
    torch::Tensor ball_meta_i32,
    torch::Tensor cluster_o,
    torch::Tensor cluster_delta,
    torch::Tensor cluster_t_off,
    torch::Tensor block_bounds,
    torch::Tensor exact_vecs,
    torch::Tensor perm_to_orig_id,
    torch::Tensor probe_lists,
    torch::Tensor n_probes_arr,
    torch::Tensor op_flags_i32,
    torch::Tensor l0, torch::Tensor r0,
    torch::Tensor l1, torch::Tensor r1,
    torch::Tensor cat_mask_i64,
    torch::Tensor w0, torch::Tensor w3, torch::Tensor alg_bound,
    int64_t N_pad,
    int64_t D_pad,
    int64_t M,
    int64_t k,
    int64_t G_split,
    int64_t regime,
    int64_t sketch_mode,
    bool enable_warp_compaction,
    bool enable_tile_header,
    bool rec_tel,
    torch::Tensor g_tau_int,
    torch::Tensor partial_d,
    torch::Tensor partial_i,
    torch::Tensor out_d,
    torch::Tensor out_i,
    torch::Tensor telemetry_i64
) {
    int32_t B = (int32_t)queries.size(0);
    int32_t D = (int32_t)queries.size(1);
    int32_t nlist = (int32_t)(cluster_t_off.size(0) - 1);
    int32_t P_stride = (int32_t)probe_lists.size(1);
    if (B <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    g_tau_int.fill_(0x7f7fffff); // IEEE-754 FLT_MAX bit pattern

    dim3 grid((uint32_t)G_split, (uint32_t)B, 1);
    auto* hdr_ptr  = reinterpret_cast<const TileHeader*>(headers_u8.data_ptr<uint8_t>());
    auto* slap_ptr = reinterpret_cast<const uint64_t*>(slap_cols_i64.data_ptr<int64_t>());
    auto* i4_ptr   = reinterpret_cast<const uint4*>(int4_i32.data_ptr<int32_t>());
    auto* sg_ptr   = reinterpret_cast<const uint32_t*>(sign_i32.data_ptr<int32_t>());
    auto* bm_ptr   = reinterpret_cast<const __half2*>(ball_meta_i32.data_ptr<int32_t>());
    auto* tel_ptr  = reinterpret_cast<uint64_t*>(telemetry_i64.data_ptr<int64_t>());

    float* pd_ptr   = (G_split == 1) ? out_d.data_ptr<float>() : partial_d.data_ptr<float>();
    int64_t* pi_ptr = (G_split == 1) ? out_i.data_ptr<int64_t>() : partial_i.data_ptr<int64_t>();

    if (exact_vecs.scalar_type() == torch::kFloat32) {
        prism_search_kernel<float><<<grid, 256, 0, stream>>>(
            queries.data_ptr<float>(), hdr_ptr, slap_ptr, i4_ptr, sg_ptr, bm_ptr,
            cluster_o.data_ptr<float>(), cluster_delta.data_ptr<float>(),
            cluster_t_off.data_ptr<int32_t>(), block_bounds.data_ptr<int32_t>(),
            exact_vecs.data_ptr<float>(), perm_to_orig_id.data_ptr<int64_t>(),
            probe_lists.data_ptr<int32_t>(), n_probes_arr.data_ptr<int32_t>(),
            reinterpret_cast<const uint32_t*>(op_flags_i32.data_ptr<int32_t>()),
            l0.data_ptr<int32_t>(), r0.data_ptr<int32_t>(),
            l1.data_ptr<int32_t>(), r1.data_ptr<int32_t>(),
            reinterpret_cast<const uint64_t*>(cat_mask_i64.data_ptr<int64_t>()),
            w0.data_ptr<int32_t>(), w3.data_ptr<int32_t>(), alg_bound.data_ptr<int32_t>(),
            B, N_pad, nlist, D, (int32_t)D_pad, (int32_t)M, (int32_t)k, P_stride,
            (int32_t)G_split, (int32_t)regime, (int32_t)sketch_mode,
            enable_warp_compaction, enable_tile_header, rec_tel,
            g_tau_int.data_ptr<int32_t>(), pd_ptr, pi_ptr, tel_ptr
        );
    } else if (exact_vecs.scalar_type() == torch::kUInt8) {
        prism_search_kernel<uint8_t><<<grid, 256, 0, stream>>>(
            queries.data_ptr<float>(), hdr_ptr, slap_ptr, i4_ptr, sg_ptr, bm_ptr,
            cluster_o.data_ptr<float>(), cluster_delta.data_ptr<float>(),
            cluster_t_off.data_ptr<int32_t>(), block_bounds.data_ptr<int32_t>(),
            exact_vecs.data_ptr<uint8_t>(), perm_to_orig_id.data_ptr<int64_t>(),
            probe_lists.data_ptr<int32_t>(), n_probes_arr.data_ptr<int32_t>(),
            reinterpret_cast<const uint32_t*>(op_flags_i32.data_ptr<int32_t>()),
            l0.data_ptr<int32_t>(), r0.data_ptr<int32_t>(),
            l1.data_ptr<int32_t>(), r1.data_ptr<int32_t>(),
            reinterpret_cast<const uint64_t*>(cat_mask_i64.data_ptr<int64_t>()),
            w0.data_ptr<int32_t>(), w3.data_ptr<int32_t>(), alg_bound.data_ptr<int32_t>(),
            B, N_pad, nlist, D, (int32_t)D_pad, (int32_t)M, (int32_t)k, P_stride,
            (int32_t)G_split, (int32_t)regime, (int32_t)sketch_mode,
            enable_warp_compaction, enable_tile_header, rec_tel,
            g_tau_int.data_ptr<int32_t>(), pd_ptr, pi_ptr, tel_ptr
        );
    } else {
        TORCH_CHECK(false, "Unsupported exact_vecs dtype");
    }

    if (G_split > 1) {
        int threads = 128;
        int blocks  = (B + threads - 1) / threads;
        merge_partial_topk_kernel<<<blocks, threads, 0, stream>>>(
            partial_d.data_ptr<float>(), partial_i.data_ptr<int64_t>(),
            B, (int32_t)G_split, (int32_t)k,
            out_d.data_ptr<float>(), out_i.data_ptr<int64_t>()
        );
    }
}

__global__ void mark_deleted_rows_kernel(
    TileHeader* __restrict__ headers,
    uint64_t* __restrict__ slap_cols,
    const int64_t* __restrict__ pad_rows,
    int64_t n_del,
    int64_t N_pad
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_del) return;
    int64_t r = pad_rows[i];
    if (r < 0 || r >= N_pad) return;
    atomicOr((unsigned long long*)&slap_cols[r], (unsigned long long)(1ULL << 63));
    int64_t wt = r >> 5;
    // Set bit 0 of TileHeader::flags (offset 14 bytes = 7 uint16s -> word 3 high 16 bits in uint32)
    uint32_t* hdr_u32 = reinterpret_cast<uint32_t*>(&headers[wt]);
    atomicOr(&hdr_u32[3], (1u << 16));
}

void mark_deleted_rows(
    torch::Tensor headers_u8,
    torch::Tensor slap_cols_i64,
    torch::Tensor pad_rows_i64,
    int64_t N_pad
) {
    int64_t n_del = pad_rows_i64.size(0);
    if (n_del <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    int blocks  = (int)((n_del + threads - 1) / threads);
    mark_deleted_rows_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<TileHeader*>(headers_u8.data_ptr<uint8_t>()),
        reinterpret_cast<uint64_t*>(slap_cols_i64.data_ptr<int64_t>()),
        pad_rows_i64.data_ptr<int64_t>(),
        n_del,
        N_pad
    );
}

} // namespace prism

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_prism_tiles", &prism::encode_prism_tiles, "Encode PRISM SLAP columns & Sign-Refined Subspace Ball tiles");
    m.def("encode_prism_sparse_tiles", &prism::encode_prism_sparse_tiles, "Incremental sparse-tile GPU encoder");
    m.def("estimate_query_plan", &prism::estimate_query_plan, "Fast GPU TileHeader selectivity & cluster prewarm planner");
    m.def("launch_prism_search", &prism::launch_prism_search, "Fused PRISM multi-regime GPU search kernel");
    m.def("mark_deleted_rows", &prism::mark_deleted_rows, "O(1) GPU tombstone deletion via SLAP & TileHeader flags");
}
