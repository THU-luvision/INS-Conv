// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Rulebook Format
// rules[0][0] == mode
// rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
// rules[0][2] == nInputRows
// rules[0][3] == nOutputRows
// rules[1]   nOutputRows x (1+maxActive)

#include <cstdint>
#include <easy/profiler.h>

template <typename T>
__global__ void InputLayer_fp_(T *input_features, T *output_features, Int nRows,
                               Int maxActive, Int nPlanes, Int *rules,
                               bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
    for (int i = 1; i <= nActive; i++) {
      T *inp = input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        out[plane] += multiplier * inp[plane];
    }
  }
}


template <typename T>
void InputLayer_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average) {
  cudaMemcpy(rules_gpu, rules_cpu, sizeof(Int) * nRows * (1 + maxActive),
             cudaMemcpyHostToDevice);
  InputLayer_fp_<
      T><<<std::min(nRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(
      input_features, output_features, nRows, maxActive, nPlanes, rules_gpu,
      average);
}

template <typename T>
__global__ void InputLayer_bp_(T *d_input_features, T *d_output_features,
                               Int nRows, Int maxActive, Int nPlanes,
                               Int *rules, bool average) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = d_output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T multiplier = (average and nActive > 0) ? (T)1 / nActive : (T)1;
    for (int i = 1; i <= nActive; i++) {
      T *inp = d_input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        atomicAdd(&inp[plane], multiplier * out[plane]);
    }
  }
}

template <typename T>
void InputLayer_bp(T *d_input_features, T *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average) {

  EASY_FUNCTION(profiler::colors::Blue200);  
  
  cudaMemcpy(rules_gpu, rules_cpu, sizeof(Int) * nRows * (1 + maxActive),
             cudaMemcpyHostToDevice);
  InputLayer_bp_<T><<<std::min(nRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(
      d_input_features, d_output_features, nRows, maxActive, nPlanes, rules_gpu,
      average);
}



/*--------------------------------------------------*/
template <typename T>
__global__ void InputLayer_inc_fp_(T *input_features, T *output_features, Int nRows,
                               Int maxActive, Int nPlanes, Int *rules,
                               bool average, uint32_t *pre_exist, T *pre_output_feats, 
                               Int *pre_rules, Int pre_maxActive, Int pre_nRows) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int *pre_r = pre_exist[row] != 0xFFFFFFFF ? pre_rules + pre_exist[row] * (1 + pre_maxActive) : NULL;
    Int nActive = r[0];
    Int pre_nActive = pre_r == NULL ? 0: pre_r[0];
    T multiplier = (average and nActive > 0) ? (T)1 / (nActive + pre_nActive) : (T)1;
    for (int i = 1; i <= nActive; i++) {
      T *inp = input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x)
        out[plane] += multiplier * inp[plane];
    }
    if (pre_r != NULL) {
      T * pre_feats_loc = pre_output_feats + pre_exist[row] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x) {
        out[plane] = out[plane] + pre_feats_loc[plane] * (pre_nActive * multiplier - 1); 
      }
    }
  }
}

template <typename T>
void InputLayer_inc_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, uint32_t *pre_exist, T *pre_output_feats, Int *pre_rules_cpu, 
                   Int *pre_rules_gpu, Int pre_nRows, Int pre_maxActive) {
  cudaMemcpy(rules_gpu, rules_cpu, sizeof(Int) * nRows * (1 + maxActive),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pre_rules_gpu, pre_rules_cpu, sizeof(Int) * pre_nRows * (1 + pre_maxActive),
             cudaMemcpyHostToDevice);
  
  InputLayer_inc_fp_<T><<<std::min(nRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(
      input_features, output_features, nRows, maxActive, nPlanes, rules_gpu,
      average, pre_exist, pre_output_feats, pre_rules_gpu, pre_maxActive, pre_nRows);
}

template <typename T>
__global__ void InputLayer_inc_bp_(T *d_input_features, T *d_output_features,
                               Int nRows, Int maxActive, Int nPlanes,
                               Int *rules, bool average, Int *pre_exist, T* pre_input_feats) {
  for (int row = blockIdx.x; row < nRows; row += gridDim.x) {
    T *out = d_output_features + row * nPlanes;
    Int *r = rules + row * (1 + maxActive);
    Int nActive = r[0];
    T* pre_input_loc = pre_input_feats + pre_exist[row] * nPlanes;
    for (int i = 1; i <= nActive; i++) {
      T *inp = d_input_features + r[i] * nPlanes;
      for (Int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x) {
        atomicAdd(&inp[plane],  out[plane]);
        if (pre_exist[row] >= 0) {
          atomicAdd(&inp[plane], pre_input_loc[plane]);
        }
      }
    }
  }
}

template <typename T>
void InputLayer_inc_bp(T *d_input_features, T *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, Int *pre_exist, T *pre_input_feats) {

  EASY_FUNCTION(profiler::colors::Blue200);  
  
  cudaMemcpy(rules_gpu, rules_cpu, sizeof(Int) * nRows * (1 + maxActive),
             cudaMemcpyHostToDevice);
  InputLayer_inc_bp_<T><<<std::min(nRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(
      d_input_features, d_output_features, nRows, maxActive, nPlanes, rules_gpu,
      average, pre_exist, pre_input_feats);
}