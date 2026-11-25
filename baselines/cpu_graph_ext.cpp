#include <torch/extension.h>
#include <omp.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>

namespace cpu_graph {

constexpr uint32_t PRED_COL0_RANGE = 1u;
constexpr uint32_t PRED_COL1_RANGE = 2u;
constexpr uint32_t PRED_COL2_CAT   = 4u;
constexpr uint32_t PRED_ALG_EXPR   = 8u;

inline bool eval_slap_cpu(
    uint64_t packed,
    uint32_t op_flags,
    int32_t l0, int32_t r0,
    int32_t l1, int32_t r1,
    uint64_t cat_mask,
    int32_t w0, int32_t w3, int32_t alg_bound
) {
    int32_t c0 = (int32_t)(packed & 0xFFFFULL);
    int32_t c1 = (int32_t)((packed >> 16) & 0xFFFFULL);
    int32_t c3 = (int32_t)((packed >> 32) & 0xFFFFULL);
    uint32_t c2 = (uint32_t)((packed >> 48) & 0x3FULL);

    if ((op_flags & PRED_COL0_RANGE) && (c0 < l0 || c0 > r0)) return false;
    if ((op_flags & PRED_COL1_RANGE) && (c1 < l1 || c1 > r1)) return false;
    if ((op_flags & PRED_COL2_CAT) && (((1ULL << c2) & cat_mask) == 0ULL)) return false;
    if ((op_flags & PRED_ALG_EXPR) && ((w0 * c0 + w3 * c3) > alg_bound)) return false;
    return true;
}

inline float l2_sq_simd(const float* __restrict__ a, const float* __restrict__ b, int D) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int j = 0; j < D; ++j) {
        float df = a[j] - b[j];
        sum += df * df;
    }
    return sum;
}

// Parallel 48-thread ACORN-gamma / HNSW-Filter search over flat CSR/fixed-degree proximity graph
void search_filtered_graph_omp(
    torch::Tensor base_vecs,        // [N, D] float32 CPU
    torch::Tensor slap_cols_i64,    // [N] int64 CPU
    torch::Tensor neighbors,        // [N, deg] int32 CPU (-1 padded)
    torch::Tensor entry_points,     // [n_entry] int32 CPU
    torch::Tensor queries,          // [B, D] float32 CPU
    torch::Tensor op_flags_i32,
    torch::Tensor l0_t, torch::Tensor r0_t,
    torch::Tensor l1_t, torch::Tensor r1_t,
    torch::Tensor cat_mask_i64,
    torch::Tensor w0_t, torch::Tensor w3_t, torch::Tensor alg_bound_t,
    int64_t k,
    int64_t ef_search,
    int64_t M_base,
    bool acorn_two_hop,
    torch::Tensor out_ids,          // [B, k] int64 CPU
    torch::Tensor out_dists         // [B, k] float32 CPU
) {
    const int64_t N = base_vecs.size(0);
    const int32_t D = (int32_t)base_vecs.size(1);
    const int32_t B = (int32_t)queries.size(0);
    const int32_t deg = (int32_t)neighbors.size(1);
    const int32_t n_ep = (int32_t)entry_points.size(0);

    const float* xb = base_vecs.data_ptr<float>();
    const uint64_t* sc = reinterpret_cast<const uint64_t*>(slap_cols_i64.data_ptr<int64_t>());
    const int32_t* adj = neighbors.data_ptr<int32_t>();
    const int32_t* eps = entry_points.data_ptr<int32_t>();
    const float* xq = queries.data_ptr<float>();

    const uint32_t* op_arr = reinterpret_cast<const uint32_t*>(op_flags_i32.data_ptr<int32_t>());
    const int32_t* l0_arr = l0_t.data_ptr<int32_t>();
    const int32_t* r0_arr = r0_t.data_ptr<int32_t>();
    const int32_t* l1_arr = l1_t.data_ptr<int32_t>();
    const int32_t* r1_arr = r1_t.data_ptr<int32_t>();
    const uint64_t* cm_arr = reinterpret_cast<const uint64_t*>(cat_mask_i64.data_ptr<int64_t>());
    const int32_t* w0_arr = w0_t.data_ptr<int32_t>();
    const int32_t* w3_arr = w3_t.data_ptr<int32_t>();
    const int32_t* ab_arr = alg_bound_t.data_ptr<int32_t>();

    int64_t* oi = out_ids.data_ptr<int64_t>();
    float* od   = out_dists.data_ptr<float>();

    #pragma omp parallel
    {
        static thread_local std::vector<uint32_t> visited_tag;
        static thread_local uint32_t cur_tag = 1u;
        if (visited_tag.size() < (size_t)N) {
            visited_tag.assign((size_t)N, 0u);
            cur_tag = 1u;
        }

        #pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < B; ++b) {
            if (++cur_tag == 0u) {
                std::fill(visited_tag.begin(), visited_tag.end(), 0u);
                cur_tag = 1u;
            }
            const float* q = xq + (int64_t)b * D;
            uint32_t op = op_arr[b];
            int32_t l0 = l0_arr[b], r0 = r0_arr[b];
            int32_t l1 = l1_arr[b], r1 = r1_arr[b];
            uint64_t cm = cm_arr[b];
            int32_t w0 = w0_arr[b], w3 = w3_arr[b], ab = ab_arr[b];

            using Pair = std::pair<float, int32_t>;
            // Min-heap of candidates to explore; Max-heap of beam & filtered results
            std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cand_q;
            std::priority_queue<Pair> nav_beam;
            std::priority_queue<Pair> res_heap;

            for (int e = 0; e < n_ep; ++e) {
                int32_t u = eps[e];
                if (u < 0 || u >= N || visited_tag[u] == cur_tag) continue;
                visited_tag[u] = cur_tag;
                float d = l2_sq_simd(q, xb + (int64_t)u * D, D);
                cand_q.emplace(d, u);
                nav_beam.emplace(d, u);
                if ((int64_t)nav_beam.size() > ef_search) nav_beam.pop();
                if (eval_slap_cpu(sc[u], op, l0, r0, l1, r1, cm, w0, w3, ab)) {
                    res_heap.emplace(d, u);
                    if ((int64_t)res_heap.size() > k) res_heap.pop();
                }
            }

            int64_t max_expansions = ef_search * 12;
            int64_t steps = 0;

            while (!cand_q.empty() && steps < max_expansions) {
                auto [du, u] = cand_q.top();
                cand_q.pop();
                float nav_bound = ((int64_t)nav_beam.size() >= ef_search) ? nav_beam.top().first : FLT_MAX;
                float res_bound = ((int64_t)res_heap.size() >= k) ? res_heap.top().first : FLT_MAX;
                if (du > std::max(nav_bound, res_bound) && (int64_t)res_heap.size() >= k) {
                    break;
                }
                steps++;

                const int32_t* nu = adj + (int64_t)u * deg;
                int pass_found = 0;
                int two_hop_budget = (int)M_base;

                for (int j = 0; j < deg; ++j) {
                    int32_t v = nu[j];
                    if (v < 0) break;
                    if (j + 1 < deg && nu[j + 1] >= 0) {
                        __builtin_prefetch(xb + (int64_t)nu[j + 1] * D, 0, 1);
                    }
                    bool v_ok = eval_slap_cpu(sc[v], op, l0, r0, l1, r1, cm, w0, w3, ab);

                    if (!acorn_two_hop) {
                        // Standard HNSW-Filter: traverse all edges, collect passing nodes
                        if (visited_tag[v] != cur_tag) {
                            visited_tag[v] = cur_tag;
                            float dv = l2_sq_simd(q, xb + (int64_t)v * D, D);
                            if ((int64_t)nav_beam.size() < ef_search || dv < nav_beam.top().first) {
                                cand_q.emplace(dv, v);
                                nav_beam.emplace(dv, v);
                                if ((int64_t)nav_beam.size() > ef_search) nav_beam.pop();
                            }
                            if (v_ok && ((int64_t)res_heap.size() < k || dv < res_heap.top().first)) {
                                res_heap.emplace(dv, v);
                                if ((int64_t)res_heap.size() > k) res_heap.pop();
                            }
                        }
                    } else {
                        // ACORN-gamma: collect up to M_base predicate-passing neighbors; if scarce, expand 2-hop!
                        if (v_ok) {
                            if (visited_tag[v] != cur_tag) {
                                visited_tag[v] = cur_tag;
                                float dv = l2_sq_simd(q, xb + (int64_t)v * D, D);
                                if ((int64_t)nav_beam.size() < ef_search || dv < nav_beam.top().first) {
                                    cand_q.emplace(dv, v);
                                    nav_beam.emplace(dv, v);
                                    if ((int64_t)nav_beam.size() > ef_search) nav_beam.pop();
                                }
                                if ((int64_t)res_heap.size() < k || dv < res_heap.top().first) {
                                    res_heap.emplace(dv, v);
                                    if ((int64_t)res_heap.size() > k) res_heap.pop();
                                }
                            }
                            if (++pass_found >= (int)M_base) break;
                        } else if (two_hop_budget > 0) {
                            // 2-hop predicate-aware expansion through failing neighbor v
                            const int32_t* nv = adj + (int64_t)v * deg;
                            for (int h = 0; h < (int)M_base && two_hop_budget > 0; ++h) {
                                int32_t w = nv[h];
                                if (w < 0) break;
                                two_hop_budget--;
                                if (visited_tag[w] == cur_tag) continue;
                                if (eval_slap_cpu(sc[w], op, l0, r0, l1, r1, cm, w0, w3, ab)) {
                                    visited_tag[w] = cur_tag;
                                    float dw = l2_sq_simd(q, xb + (int64_t)w * D, D);
                                    if ((int64_t)nav_beam.size() < ef_search || dw < nav_beam.top().first) {
                                        cand_q.emplace(dw, w);
                                        nav_beam.emplace(dw, w);
                                        if ((int64_t)nav_beam.size() > ef_search) nav_beam.pop();
                                    }
                                    if ((int64_t)res_heap.size() < k || dw < res_heap.top().first) {
                                        res_heap.emplace(dw, w);
                                        if ((int64_t)res_heap.size() > k) res_heap.pop();
                                    }
                                    if (++pass_found >= (int)M_base) break;
                                }
                            }
                        }
                    }
                }
            }

            int out_cnt = (int)res_heap.size();
            for (int s = (int)k - 1; s >= 0; --s) {
                if (s >= out_cnt) {
                    oi[(int64_t)b * k + s] = -1LL;
                    od[(int64_t)b * k + s] = FLT_MAX;
                } else {
                    auto [d_val, id_val] = res_heap.top();
                    res_heap.pop();
                    oi[(int64_t)b * k + s] = (int64_t)id_val;
                    od[(int64_t)b * k + s] = d_val;
                }
            }
        }
    }
}

} // namespace cpu_graph

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("search_filtered_graph_omp", &cpu_graph::search_filtered_graph_omp, "48-thread OpenMP ACORN-gamma & HNSW-Filter");
}
