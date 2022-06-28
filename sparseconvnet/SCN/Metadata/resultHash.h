#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <sparsehash/dense_hash_map>
#include <torch/extension.h>
#include <cassert>
#include <unordered_map>
#include "Metadata.h"
#include "algorithm"
#include "sys/time.h"
#include "google/dense_hash_map"
#include "easy/profiler.h"

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

template<Int dimension> class Result {
    public:
        int maximum;
        

        google::dense_hash_map<Point<dimension>, std::pair<Int, float>, IntArrayHash<dimension>> result;
        
        Result(float maximum=0.7): maximum(maximum) 
        {
            Point<dimension> lb;
            lb[0] = lb[1] = lb[2] = -100000000;
            result.set_empty_key(lb);
        }

        void set(torch::Tensor ps, torch::Tensor label) {
            if (ps.ndimension() == 1) {
                auto p = LongTensorToPoint<dimension>(ps);
                result[p] = std::make_pair(torch::argmax(label, -1).item().to<Int>(), 0);
            } else if (ps.ndimension() == 2) {
                for (Int i = 0; i < ps.size(0); i++) {
                    auto p = ps[i];
                    auto p_array = LongTensorToPoint<dimension>(p);
                    result[p_array] = std::make_pair(torch::argmax(label[i]).item().to<Int>(), 0);
                }
            }
        }

        void update(torch::Tensor ps, torch::Tensor label) {
            // should do tensor operation in batch, not single.
            // 30ms ~ 20000 points,  mainly because hash table.
            assert(ps.ndimension() == 2);
            EASY_FUNCTION(profiler::colors::Blue100);
            double st, end;
            
            auto prob_indexs = label.max(1);
            int64_t *indexs = std::get<1>(prob_indexs).data<int64_t>();
            float *max_probs = std::get<0>(prob_indexs).data<float>();

            static bool pre_exist[1000000];
            memset(pre_exist, 0, sizeof(bool) * ps.size(0));
            EASY_BLOCK("Allocate hash memory");
            auto ps_data = ps.data<long>();
            auto stride = ps.stride(0);
            for (int i = 0; i < ps.size(0); i++) {
                Point<dimension> x;
                x[0] = ps_data[i * stride];
                x[1] = ps_data[i * stride + 1];
                x[2] = ps_data[i * stride + 2];
                auto iter = result.find(x);
                if (iter != result.end()) pre_exist[i] = 1;
                else {
                    auto &t = result[x];
                }
            }
            EASY_END_BLOCK;
            EASY_BLOCK("Update values");
            #pragma omp parallel for 
            for (Int i = 0; i < ps.size(0); i++) {
                Point<dimension> x;
                x[0] = ps_data[i * stride];
                x[1] = ps_data[i * stride + 1];
                x[2] = ps_data[i * stride + 2];
                Int val = indexs[i];
                float max_prob = max_probs[i];
                assert(result.find(x) != result.end());
                auto &p_result = result[x];
                if (!pre_exist[i]) 
                    p_result = std::pair<Int, float>(val, max_prob);
                else {
                    if (p_result.first == val) p_result.second = std::min((float)maximum, p_result.second + max_prob);
                    else p_result.second -= max_prob;
                    if (p_result.second < 0) {
                            p_result.first = val;
                            p_result.second *= -1;
                    }
                }
            }

        }

        Int query(torch::Tensor p) {
            /*single point query*/
            assert(p.ndimension()==1);
            auto iter = result.find(LongTensorToPoint<dimension>(p));
            return iter == result.end() ? -1: (iter->second).first;
        }

        torch::Tensor multiQuery(torch::Tensor ps) {
            auto label = torch::zeros(ps.size(0), torch::dtype(torch::kLong));
            for (Int i = 0; i < ps.size(0); i++) {
                auto p = ps[i];
                auto iter = result.find(LongTensorToPoint<dimension>(p));
                label[i] = iter == result.end() ? -1 : (iter->second).first;
            }
            return label;
        }
        
        torch::Tensor get_all_points_labels() {
            auto ret = torch::zeros({Int(result.size()), dimension + 1}, torch::dtype(torch::kLong));
            int i = 0;

            for (auto& p : result) {
                auto t = torch::zeros({dimension+1}, torch::dtype(torch::kLong));
                for (Int j = 0; j < dimension; j++) t[j] = p.first[j];
                t[dimension] = p.second.first;
                ret[i] = t;
                i++;
            }
            return ret;
        }

        void finalize(torch::Tensor points) {
            std::unordered_map<Point<dimension>, int, IntArrayHash<dimension>> vis;
            for (int i = 0; i < points.size(0); i++) {
                auto p = points[i];
                vis[LongTensorToPoint<dimension>(p)] = 1;
            }
            for (auto it = result.begin(); it != result.end(); ) {
                if (vis.find(it->first) == vis.end()) {
                    result.erase(it++);
                } else it++;
            }
        }

        void clear() {
            result.clear();
        }
        
};


template<Int dimension> class ResultOfWeight {
    public:
        std::unordered_map<Point<dimension>, torch::Tensor, IntArrayHash<dimension>> result;
        ResultOfWeight(){}

        void update(torch::Tensor ps, torch::Tensor label) {
            assert(ps.ndimension() == 2);
            for (Int i = 0; i < ps.size(0); i++) {
                auto p = ps[i];
                auto p_array = LongTensorToPoint<dimension>(p);
                auto iter = result.find(p_array);
                if (iter == result.end()) 
                    result[p_array] = label[i];
                else {
                    result[p_array] = result[p_array] * 0.5 + label[i];
                }
            }
        }

        Int query(torch::Tensor p) {
            /*single point query*/
            assert(p.ndimension()==1);
            auto iter = result.find(LongTensorToPoint<dimension>(p));
            return iter == result.end() ? -1: (iter->second).argmax(-1).template item<int>();
        }

        torch::Tensor multiQuery(torch::Tensor ps) {
            auto label = torch::zeros(ps.size(0), torch::dtype(torch::kLong));
            for (Int i = 0; i < ps.size(0); i++) {
                auto p = ps[i];
                auto iter = result.find(LongTensorToPoint<dimension>(p));
                label[i] = iter == result.end() ? -1 : (iter->second).argmax(-1).template item<int>();
            }
            return label;
        }
        
        torch::Tensor get_all_points_labels() {
            auto ret = torch::zeros({Int(result.size()), dimension + 1}, torch::dtype(torch::kLong));
            int i = 0;

            for (auto& p : result) {
                auto t = torch::zeros({dimension+1}, torch::dtype(torch::kLong));
                for (Int j = 0; j < dimension; j++) t[j] = p.first[j];
                t[dimension] = torch::argmax(p.second, -1);
                ret[i] = t;
                i++;
            }
            return ret;
        }

        void finalize(torch::Tensor points) {
            std::unordered_map<Point<dimension>, int, IntArrayHash<dimension>> vis;
            for (int i = 0; i < points.size(0); i++) {
                auto p = points[i];
                vis[LongTensorToPoint<dimension>(p)] = 1;
            }
            for (auto it = result.begin(); it != result.end(); ) {
                if (vis.find(it->first) == vis.end()) {
                    result.erase(it++);
                } else it++;
            }
        }

        void clear() {
            result.clear();
        }
};