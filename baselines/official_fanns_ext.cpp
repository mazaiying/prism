// Official Open-Source Filtered ANNS Bindings:
//   1. Official ACORN-gamma (`guestrin-lab/ACORN` — `faiss::IndexACORNFlat`, SIGMOD'24)
//   2. Official SeRF 2D Segment Graph (`rutgers-db/SeRF` — `SeRF::IndexSegmentGraph2D`, SIGMOD'24)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <numeric>
#include <vector>

#include <faiss/IndexACORN.h>
#ifdef debug
#undef debug
#endif

// Include official SeRF headers after undefining ACORN's `debug` macro
#include "data_wrapper.h"
#include "index_base.h"
#include "segment_graph_2d.h"

namespace py = pybind11;

class OfficialACORNWrapper {
public:
    int d_;
    int M_;
    int gamma_;
    int M_beta_;
    int64_t N_ = 0;
    std::vector<int> metadata_;
    std::unique_ptr<faiss::IndexACORNFlat> index_;

    OfficialACORNWrapper(int d, int M, int gamma, py::array_t<int32_t, py::array::c_style | py::array::forcecast> meta_arr, int M_beta)
        : d_(d), M_(M), gamma_(gamma), M_beta_(M_beta) {
        auto buf = meta_arr.request();
        N_ = buf.shape[0];
        const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
        metadata_.assign(ptr, ptr + N_);
        index_ = std::make_unique<faiss::IndexACORNFlat>(d_, M_, gamma_, metadata_, M_beta_, faiss::METRIC_L2);
    }

    void build(py::array_t<float, py::array::c_style | py::array::forcecast> xb, int ef_construction, int num_threads) {
        auto buf = xb.request();
        int64_t n = buf.shape[0];
        const float* x_ptr = static_cast<const float*>(buf.ptr);
        omp_set_num_threads(num_threads);
        index_->acorn.efConstruction = ef_construction;
        py::gil_scoped_release release;
        index_->add(n, x_ptr);
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>> search(
        py::array_t<float, py::array::c_style | py::array::forcecast> xq,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> filter_map,
        int k,
        int ef_search,
        int num_threads
    ) {
        auto q_buf = xq.request();
        auto f_buf = filter_map.request();
        int64_t nq = q_buf.shape[0];
        const float* q_ptr = static_cast<const float*>(q_buf.ptr);
        char* f_ptr = reinterpret_cast<char*>(f_buf.ptr);

        py::array_t<int64_t> labels({nq, static_cast<int64_t>(k)});
        py::array_t<float> dists({nq, static_cast<int64_t>(k)});
        faiss::idx_t* l_ptr = reinterpret_cast<faiss::idx_t*>(labels.mutable_data());
        float* d_ptr = dists.mutable_data();

        omp_set_num_threads(num_threads);
        index_->acorn.efSearch = ef_search;
        {
            py::gil_scoped_release release;
            index_->search(nq, q_ptr, k, d_ptr, l_ptr, f_ptr);
        }
        return {labels, dists};
    }
};

class OfficialSeRF2DWrapper {
public:
    int d_;
    int N_;
    std::vector<int32_t> sorted_to_orig_;
    std::vector<float> sorted_keys_;
    std::unique_ptr<DataWrapper> dw_;
    std::unique_ptr<base_hnsw::L2Space> space_;
    std::unique_ptr<BaseIndex::IndexParams> iparams_;
    std::unique_ptr<SeRF::IndexSegmentGraph2D> index_;

    OfficialSeRF2DWrapper(
        py::array_t<float, py::array::c_style | py::array::forcecast> xb,
        py::array_t<float, py::array::c_style | py::array::forcecast> attr_col,
        int index_k = 16,
        int ef_construction = 100,
        int ef_max = 400,
        int num_threads = 48
    ) {
        auto x_buf = xb.request();
        auto a_buf = attr_col.request();
        N_ = static_cast<int>(x_buf.shape[0]);
        d_ = static_cast<int>(x_buf.shape[1]);
        const float* x_ptr = static_cast<const float*>(x_buf.ptr);
        const float* a_ptr = static_cast<const float*>(a_buf.ptr);

        sorted_to_orig_.resize(N_);
        std::iota(sorted_to_orig_.begin(), sorted_to_orig_.end(), 0);
        std::stable_sort(sorted_to_orig_.begin(), sorted_to_orig_.end(), [&](int32_t i, int32_t j) {
            return a_ptr[i] < a_ptr[j];
        });

        sorted_keys_.resize(N_);
        dw_ = std::make_unique<DataWrapper>(1, 10, "prism_bench", N_);
        dw_->version = "Benchmark";
        dw_->data_dim = static_cast<size_t>(d_);
        dw_->nodes.resize(N_);
        dw_->nodes_keys.resize(N_);
        for (int pos = 0; pos < N_; ++pos) {
            int32_t orig = sorted_to_orig_[pos];
            sorted_keys_[pos] = a_ptr[orig];
            dw_->nodes_keys[pos] = pos;
            const float* row = x_ptr + static_cast<size_t>(orig) * d_;
            dw_->nodes[pos].assign(row, row + d_);
        }

        space_ = std::make_unique<base_hnsw::L2Space>(static_cast<size_t>(d_));
        iparams_ = std::make_unique<BaseIndex::IndexParams>(
            static_cast<unsigned>(index_k),
            static_cast<unsigned>(ef_construction),
            static_cast<unsigned>(ef_construction),
            static_cast<unsigned>(ef_max)
        );
        iparams_->recursion_type = BaseIndex::IndexParams::MAX_POS;
        index_ = std::make_unique<SeRF::IndexSegmentGraph2D>(space_.get(), dw_.get());
        index_->isLog = false;

        omp_set_num_threads(num_threads);
        py::gil_scoped_release release;
        index_->buildIndex(iparams_.get());
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>> search_range(
        py::array_t<float, py::array::c_style | py::array::forcecast> xq,
        py::array_t<float, py::array::c_style | py::array::forcecast> lo_vals,
        py::array_t<float, py::array::c_style | py::array::forcecast> hi_vals,
        int k,
        int ef_search,
        int num_threads = 48
    ) {
        auto q_buf = xq.request();
        int nq = static_cast<int>(q_buf.shape[0]);
        const float* q_ptr = static_cast<const float*>(q_buf.ptr);
        const float* lo_ptr = static_cast<const float*>(lo_vals.request().ptr);
        const float* hi_ptr = static_cast<const float*>(hi_vals.request().ptr);

        py::array_t<int64_t> out_ids({static_cast<int64_t>(nq), static_cast<int64_t>(k)});
        py::array_t<float> out_dists({static_cast<int64_t>(nq), static_cast<int64_t>(k)});
        int64_t* id_ptr = out_ids.mutable_data();
        float* dist_ptr = out_dists.mutable_data();

        omp_set_num_threads(num_threads);
        {
            py::gil_scoped_release release;
            #pragma omp parallel
            {
                BaseIndex::SearchInfo sinfo(dw_.get(), iparams_.get(), "SeRF_2D", "bench");
                #pragma omp for schedule(dynamic, 1)
                for (int qi = 0; qi < nq; ++qi) {
                    for (int j = 0; j < k; ++j) {
                        id_ptr[qi * k + j] = -1;
                        dist_ptr[qi * k + j] = 1e30f;
                    }
                    auto it_lo = std::lower_bound(sorted_keys_.begin(), sorted_keys_.end(), lo_ptr[qi]);
                    auto it_hi = std::upper_bound(sorted_keys_.begin(), sorted_keys_.end(), hi_ptr[qi]);
                    int pos_lo = static_cast<int>(it_lo - sorted_keys_.begin());
                    int pos_hi = static_cast<int>(it_hi - sorted_keys_.begin()) - 1;
                    if (pos_lo > pos_hi || pos_lo >= N_ || pos_hi < 0) {
                        continue;
                    }
                    std::vector<float> qvec(q_ptr + static_cast<size_t>(qi) * d_, q_ptr + static_cast<size_t>(qi + 1) * d_);
                    BaseIndex::SearchParams sp;
                    sp.query_K = static_cast<unsigned>(k);
                    sp.search_ef = static_cast<unsigned>(ef_search);
                    sp.query_range = static_cast<unsigned>(pos_hi - pos_lo + 1);
                    std::vector<int> res = index_->rangeFilteringSearchOutBound(&sp, &sinfo, qvec, {pos_lo, pos_hi});
                    std::vector<std::pair<float, int64_t>> cand;
                    cand.reserve(res.size());
                    for (int pos : res) {
                        if (pos >= pos_lo && pos <= pos_hi) {
                            const float* v = dw_->nodes[pos].data();
                            float d2 = 0.0f;
                            for (int dim = 0; dim < d_; ++dim) {
                                float diff = qvec[dim] - v[dim];
                                d2 += diff * diff;
                            }
                            cand.emplace_back(d2, static_cast<int64_t>(sorted_to_orig_[pos]));
                        }
                    }
                    std::sort(cand.begin(), cand.end());
                    int take = std::min(k, static_cast<int>(cand.size()));
                    for (int j = 0; j < take; ++j) {
                        dist_ptr[qi * k + j] = cand[j].first;
                        id_ptr[qi * k + j] = cand[j].second;
                    }
                }
            }
        }
        return {out_ids, out_dists};
    }
};

PYBIND11_MODULE(official_fanns_ext, m) {
    m.doc() = "Official guestrin-lab/ACORN and rutgers-db/SeRF pybind11 bindings";

    py::class_<OfficialACORNWrapper>(m, "OfficialACORNIndex")
        .def(py::init<int, int, int, py::array_t<int32_t, py::array::c_style | py::array::forcecast>, int>(),
             py::arg("d"), py::arg("M") = 32, py::arg("gamma") = 12, py::arg("metadata"), py::arg("M_beta") = 64)
        .def("build", &OfficialACORNWrapper::build, py::arg("xb"), py::arg("ef_construction") = 40, py::arg("num_threads") = 48)
        .def("search", &OfficialACORNWrapper::search, py::arg("xq"), py::arg("filter_map"), py::arg("k") = 10, py::arg("ef_search") = 64, py::arg("num_threads") = 48);

    py::class_<OfficialSeRF2DWrapper>(m, "OfficialSeRF2DIndex")
        .def(py::init<py::array_t<float, py::array::c_style | py::array::forcecast>,
                      py::array_t<float, py::array::c_style | py::array::forcecast>,
                      int, int, int, int>(),
             py::arg("xb"), py::arg("attr_col"), py::arg("index_k") = 16, py::arg("ef_construction") = 100, py::arg("ef_max") = 400, py::arg("num_threads") = 48)
        .def("search_range", &OfficialSeRF2DWrapper::search_range,
             py::arg("xq"), py::arg("lo_vals"), py::arg("hi_vals"), py::arg("k") = 10, py::arg("ef_search") = 64, py::arg("num_threads") = 48);
}
