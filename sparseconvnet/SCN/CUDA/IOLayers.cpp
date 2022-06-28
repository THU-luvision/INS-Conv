// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "../Metadata/Metadata.h"
#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <cstdint>


template <typename T>
void InputLayer_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average);

template <typename T>
void InputLayer_bp(T *d_input_features, T *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average);

template <typename T>
void InputLayer_inc_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, uint32_t *pre_exist, T *pre_output_feats, Int *pre_rules_cpu, 
                   Int *pre_rules_gpu, Int pre_nRows, Int pre_maxActive);

template <typename T>
void InputLayer_inc_bp(T *d_input_features, T *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, Int *pre_exist, T *pre_input_feats);

template <typename T, Int Dimension>
void cuda_InputLayer_updateOutput(Metadata<Dimension> &m,
                                  /*long*/ at::Tensor spatialSize,
                                  /*long*/ at::Tensor input_coords,
                                  /*cuda float*/ at::Tensor input_features,
                                  /*cuda float*/ at::Tensor output_features,
                                  long batchSize, long mode) {
                                    
  EASY_FUNCTION(profiler::colors::Blue100);  

  m.inputLayer(spatialSize, input_coords, batchSize, mode);


  Int nPlanes = input_features.size(1);
  auto &rules = m.inputLayerRuleBook;

  Int maxActive = rules[0][1];
  Int nRows = rules[0][3];

  auto p = LongTensorToPoint<Dimension>(spatialSize);

  #if 0
  // auto input_normal_size = input_normal.sizes();
  Int use_normal = 0;
  if(input_normal.ndimension() == 2 && input_normal.sizes()[0] == input_features.sizes()[0]) use_normal = 1;
  if(use_normal == 1 && normal.empty())
  {
    float *input_normal_ptr = input_normal.data<float>();
    normal = std::vector<Float3>(nRows);
    Int * rule = &rules[1][0];
    for(int i = 0; i < nRows; i++)
    {
        int cnt = rule[(maxActive + 1) * i + 0];
        normal[i].x = 0;
        normal[i].y = 0;
        normal[i].z = 0;
        for(int k = 1; k <= cnt ; k++)
        {
            float *input_n_ptr = &input_normal_ptr[rule[(maxActive + 1) * i + k] * 3];
            normal[i] += Float3(input_n_ptr[0],input_n_ptr[1],input_n_ptr[2]);
        }
        if(cnt > 0)
        {
            normal[i] /= cnt;
        }
        normal[i].normalize();
    }
  }
  #endif
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({*m.inputNActive, nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_fp<T>(iF, oF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     mode == 4);
  }
}
template <typename T, Int Dimension>
void cuda_InputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features) {

  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][3];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
  } else {
    d_input_features.resize_({rules[0][2], nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_bp(diF, doF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                  mode == 4);
  }
}

template <typename T, Int Dimension>
void cuda_OutputLayer_updateOutput(Metadata<Dimension> &m,
                                   /*cuda float*/ at::Tensor input_features,
                                   /*cuda float*/ at::Tensor output_features) {

  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({rules[0][2], nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_bp(oF, iF, nRows, maxActive, nPlanes, &rules[1][0], rb, false);
  }
}
template <typename T, Int Dimension>
void cuda_OutputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features) {

  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
  } else {
    d_input_features.resize_({nRows, nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_fp<T>(doF, diF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     false);
  }
}

template <typename T, Int Dimension>
void cuda_BLInputLayer_updateOutput(Metadata<Dimension> &m,
                                    /*long*/ at::Tensor spatialSize,
                                    /*long*/ at::Tensor input_coords,
                                    /*cuda float*/ at::Tensor input_features,
                                    /*cuda float*/ at::Tensor output_features,
                                    long mode) {

  m.blLayer(spatialSize, input_coords, mode);
  Int nPlanes = input_features.size(2);
  output_features.resize_({*m.inputNActive, nPlanes});
  output_features.zero_();
  auto &rules = m.blLayerRuleBook;
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];

  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
    output_features.resize_({*m.inputNActive, nPlanes});
  } else {
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_fp<T>(iF, oF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     mode == 4);
  }
}
template <typename T, Int Dimension>
void cuda_BLInputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features) {

  auto &rules = m.blLayerRuleBook;
  Int nPlanes = d_output_features.size(1);
  Int mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];

  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
    d_input_features.resize_({rules[0][2], rules[0][3], nPlanes});
  } else {
    d_input_features.resize_({rules[0][2], rules[0][3], nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_bp(diF, doF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                  mode == 4);
  }
}

template <typename T, Int Dimension>
void cuda_BLOutputLayer_updateOutput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features) {

  auto &rules = m.blLayerRuleBook;
  Int nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
    output_features.resize_({rules[0][2], rules[0][3], nPlanes});
  } else {
    output_features.resize_({rules[0][2], rules[0][3], nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_bp(oF, iF, nRows, maxActive, nPlanes, &rules[1][0], rb, false);
  }
}
template <typename T, Int Dimension>
void cuda_BLOutputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features) {

  auto &rules = m.blLayerRuleBook;
  Int nPlanes = d_output_features.size(2);
  Int mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
    d_input_features.resize_({nRows, nPlanes});
  } else {
    d_input_features.resize_({nRows, nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_fp<T>(doF, diF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     false);
  }
}


/*--------------------------------------------------------------------------*/

template <typename T>
void InputLayer_inc_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, Int *pre_exist, T *pre_output_feats, Int *pre_rules_cpu, 
                   Int *pre_rules_gpu, Int pre_nRows, Int pre_maxActive);

template <typename T, Int Dimension>
void cuda_Incre_InputLayer_updateOutput(Metadata<Dimension> &m,
                                  /*long*/ at::Tensor spatialSize,
                                  /*long*/ at::Tensor input_coords,
                                  /*cuda float*/ at::Tensor input_features,
                                  /*cuda float*/ at::Tensor output_features,
                                  long batchSize, long mode,
                                  Metadata<Dimension>& pre_m, at::Tensor pre_output_feats) {
                                    
  EASY_FUNCTION(profiler::colors::Blue100);  

  m.inputLayer(spatialSize, input_coords, batchSize, mode);


  Int nPlanes = input_features.size(1);
  auto &rules = m.inputLayerRuleBook;
  auto &pre_rules = pre_m.inputLayerRuleBook;

  Int maxActive = rules[0][1];
  Int nRows = rules[0][3];

  Int pre_maxActive = pre_rules[0][1];
  Int pre_nRows = pre_rules[0][3];

  auto p = LongTensorToPoint<Dimension>(spatialSize);
  Int *out_points = (*m.GPU_inputSGs)[0].pHash->getAllPoints();
  uint32_t *pre_exist = NULL;
  gpuErrchk(cudaMalloc((void**)&pre_exist, sizeof(Int)*nRows));
  (*pre_m.GPU_inputSGs)[0].pHash->retrieve((uint32_t*) out_points, pre_exist, nRows);

  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({*m.inputNActive, nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    auto pre_rulesBuffer = at::empty({(int)pre_rules[1].size()}, at::CUDA(at_kINT));
    InputLayer_inc_fp<T>(iF, oF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     mode == 4, pre_exist, pre_output_feats.data<T>(), &pre_rules[1][0], 
                     pre_rulesBuffer.data<Int>(), pre_nRows, pre_maxActive);
  }

  gpuErrchk(cudaFree(pre_exist));
}

template <typename T, Int Dimension> 
void cuda_Incre_OutputLayer_updateOutput(Metadata<Dimension> &m,
                                   /*cuda float*/ at::Tensor input_features,
                                   /*cuda float*/ at::Tensor output_features,
                                   Metadata<Dimension> &pre_m, at::Tensor pre_input_feats) {
  EASY_FUNCTION(profiler::colors::Amber100);
  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  Int *out_points = (*m.GPU_inputSGs)[0].pHash->getAllPoints();
  Int *pre_exist = NULL;
  gpuErrchk(cudaMalloc((void**)&pre_exist, sizeof(Int)*nRows));
  (*pre_m.GPU_inputSGs)[0].pHash->retrieve((uint32_t*) out_points, (uint32_t*) pre_exist, nRows);

  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({rules[0][2], nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    Int *rb = rulesBuffer.data<Int>();
    InputLayer_inc_bp(oF, iF, nRows, maxActive, nPlanes, &rules[1][0], rb, false, pre_exist, pre_input_feats.data<T>());
  }
  gpuErrchk(cudaFree(pre_exist));
}