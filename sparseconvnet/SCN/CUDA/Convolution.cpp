// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "../Metadata/Metadata.h"
#include <ATen/core/Tensor.h>
#include <cstdio>
template <typename T>
void Convolution_fp_bias(T *oF, T *b, Int nPlanes, Int nActive);
template <typename T>
void Convolution_bp_bias(T *d_oF, T *d_b, Int nPlanes, Int nActive);

template <typename T>
double dConvolution_forward2(T *inFeatures, T *outFeatures, T *w,
                             RuleBook _rules, Int input_nPlanes,
                             Int input_stride, Int output_nPlanes,
                             Int output_stride);

template <typename T>
void dConvolution_backward_dW2_chunkbased(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                        RBChunkPointerList& new_rbChunkList,RuleBook &_rules, Int inputFeatureSize,
                                        T *w, T *dw, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes,
                                        Int output_stride);
template <typename T>
double dConvolution_forward2_chunkbased(T *inFeatures, T *outFeatures, T *w, Int outFeatureNum,
                             RBChunkPointerList& new_rbChunkList, Int input_nPlanes,
                             Int input_stride, Int output_nPlanes,
                             Int output_stride);

template <typename T>
double dConvolution_incre_forward2_chunkbased(T *inFeatures, T *outFeatures, T *w, Int outFeatureNum,
                             RBChunkPointerList& new_rbChunkList, Int input_nPlanes,
                             Int input_stride, Int output_nPlanes,
                             Int output_stride);

template <typename T>
void dConvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                               T *w, T *dw, RuleBook _rules, Int input_nPlanes,
                               Int input_stride, Int output_nPlanes,
                               Int output_stride);


template <typename T>
double dConvolution_incre_forward2(T *inFeatures, T *outFeatures, T *w,
                                         RuleBook &_rules, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes, Int output_stride);


template <typename T, Int Dimension>
double cuda_Convolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    output_features.resize_({nActiveOut, op});
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
double cuda_Incre_Convolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    output_features.resize_({nActiveOut, op});
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}


template <typename T, Int Dimension>
void cuda_Convolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActiveIn = m.getNActive(inputSize);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    d_input_features.resize_({nActiveIn, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}

template <typename T, Int Dimension>
double cuda_SubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias, int dilated_rate = 1) {
  EASY_FUNCTION(profiler::colors::Magenta);
  Int nActive = m.getNActive(inputSize);
EASY_BLOCK("activate");
  if (nActive) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    EASY_BLOCK("resize");
    output_features.resize_({nActive, op});
    EASY_END_BLOCK;
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActive);
    else
      output_features.zero_();
  EASY_END_BLOCK;

#ifdef SUBMANIFOLD_CHUCK


    // printf("valid chunks/points: %ld / %d %d %d\r\n", rbChunkTensors.size(), inputCnt,outputCnt, maxAddressNum);

    // dictionary is at most 100MB, could be neglected
    // suppose that rbChunkTensors should be generated on GPU


    double flops = 0;
    if (ip % 8 != 0 or op % 8 != 0) {
      EASY_BLOCK(" getSubmanifoldRuleBook rules");
      auto &_rules =  m.getSubmanifoldRuleBook(inputSize, filterSize, true,1);
      EASY_END_BLOCK;

      EASY_BLOCK("forward");
      flops += dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
      EASY_END_BLOCK;
    } else {
      EASY_BLOCK(" getSubmanifoldChunkRuleBook rules");
      auto &rbChunkTensors = m.getSubmanifoldChunkRuleBook(inputSize, filterSize, true, 1);
      EASY_END_BLOCK;

      EASY_BLOCK(" chunk forward");
      flops = dConvolution_forward2_chunkbased<T>(iF, oF, w, output_features.size(0), rbChunkTensors, ip, ip, op, op);
      EASY_END_BLOCK;
    }


    return flops;
#else
    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
#endif
  } else {
    return 0;
  }
}


template <typename T, Int Dimension>
double cuda_Incre_SubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias,
     Metadata<Dimension> &pre_m, at::Tensor pre_input_feats) {
  EASY_FUNCTION(profiler::colors::Magenta);
  
  Int nActive = m.getNActive(inputSize);
EASY_BLOCK("activate");
  if (nActive) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    EASY_BLOCK("resize");
    output_features.resize_({nActive, op});
    EASY_END_BLOCK;
    auto iF = input_features.data<T>();
    auto pre_iF = pre_input_feats.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActive);
    else
      output_features.zero_();
  EASY_END_BLOCK;
#ifdef SUBMANIFOLD_CHUCK

    // printf("valid chunks/points: %ld / %d %d %d\r\n", rbChunkTensors.size(), inputCnt,outputCnt, maxAddressNum);

    // dictionary is at most 100MB, could be neglected
    // suppose that rbChunkTensors should be generated on GPU
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    double flops = 0;

    if (ip % 8 != 0 or op % 8 != 0) {
      EASY_BLOCK(" getSubmanifoldRuleBook rules");
      auto &_rules =  m.getSubmanifoldRuleBook(inputSize, filterSize, true,1);
      EASY_END_BLOCK;
      EASY_BLOCK(" get incre SubmanifoldRuleBook rules");
      auto &pre_rules = m.getIncreSubmanifoldRuleBook(inputSize, filterSize, true, 1, pre_m);
      EASY_END_BLOCK;

      EASY_BLOCK("forward");

      flops += dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
      EASY_END_BLOCK;
      EASY_BLOCK("incre forward");
      flops += dConvolution_incre_forward2<T>(pre_iF, oF, w, pre_rules, ip, ip, op, op);
      EASY_END_BLOCK;
    } else {
      EASY_BLOCK(" getSubmanifoldChunkRuleBook rules");
      auto &rbChunkTensors = m.getSubmanifoldChunkRuleBook(inputSize, filterSize, true, 1);
      EASY_END_BLOCK;
      EASY_BLOCK(" get incre SubmanifoldChunkRuleBook rules");
      auto &pre_rbChunkTensors = m.getIncreSubmanifoldChunkRuleBook(inputSize, filterSize, true, 1, pre_m);
      EASY_END_BLOCK;
      EASY_BLOCK(" chunk forward");
      flops = dConvolution_forward2_chunkbased<T>(iF, oF, w, output_features.size(0), rbChunkTensors, ip, ip, op, op);
      EASY_END_BLOCK;
      EASY_BLOCK(" incre chunk forward");
      flops += dConvolution_incre_forward2_chunkbased<T>(pre_iF, oF, w, output_features.size(0),pre_rbChunkTensors, ip, ip, op, op);
      EASY_END_BLOCK;
    }
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    return flops;
#else
    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
#endif
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_SubmanifoldConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias,
    int dilated_rate) {
  EASY_FUNCTION(profiler::colors::Cyan);

  auto &_rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true, dilated_rate);
  Int nActive = m.getNActive(inputSize);

  if (nActive) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
//    printf("weight matrix: %d %d %d %d %d %d\r\n", ip, op,input_features.size(0),input_features.size(1),ip,op);
    d_input_features.resize_({nActive, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();
    clock_t start = clock();

//#ifdef SUBMANIFOLD_CHUCK
#if 1
//    printf("get chunk tensors\r\n");
    auto &rbChunkTensors = m.getSubmanifoldChunkRuleBook(inputSize,filterSize, true);
//    printf("convolution\r\n");

    dConvolution_backward_dW2_chunkbased<T>(iF, diF, doF, rbChunkTensors, _rules,nActive,w, dw, ip, ip, op, op);
#else
    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);
#endif
    clock_t end = clock();
    double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActive);
    }
  }
}


template <typename T, Int Dimension>
void cuda_SubmanifoldConvolution_backward_chunkbased(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias,
    int dilated_rate) {

  auto &rbChunkTensors = m.getSubmanifoldChunkRuleBook(inputSize,filterSize, true);
  auto &_rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true, dilated_rate);
  Int nActive = m.getNActive(inputSize);
//  printf("active points in dilated convolution: %d\r\n", nActive);

  if (nActive) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    d_input_features.resize_({nActive, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();
    dConvolution_backward_dW2_chunkbased<T>(iF, diF, doF, rbChunkTensors, _rules,nActive,w, dw, ip, ip, op, op);

//    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActive);
    }
  }
}


template <typename T, Int Dimension>
double cuda_PermutohedralSubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);

  if (nActive) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    output_features.resize_({nActive, op});
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActive);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_PermutohedralSubmanifoldConvolution_backward(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);

  if (nActive) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    d_input_features.resize_({nActive, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActive);
    }
  }
}

template <typename T, Int Dimension>
double cuda_FullConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {

  auto _rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActiveOut = mOut.getNActive(outputSize);
  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    output_features.resize_({nActiveOut, op});
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_FullConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActiveIn = mIn.getNActive(inputSize);
  Int nActiveOut = mOut.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    d_input_features.resize_({nActiveIn, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}
template <typename T, Int Dimension>
double cuda_RandomizedStrideConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor bias) {

  auto _rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    output_features.resize_({nActiveOut, op});
    auto iF = input_features.data<T>();
    auto oF = output_features.data<T>();
    auto w = weight.data<T>();

    if (bias.numel())
      Convolution_fp_bias(oF, bias.data<T>(), op, nActiveOut);
    else
      output_features.zero_();

    return dConvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_RandomizedStrideConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActiveIn = m.getNActive(inputSize);
  Int nActiveOut = m.getNActive(outputSize);

  if (nActiveOut) {
    Int ip = weight.size(1);
    Int op = weight.size(2);
    d_input_features.resize_({nActiveIn, ip});
    d_input_features.zero_();
    auto iF = input_features.data<T>();
    auto diF = d_input_features.data<T>();
    auto doF = d_output_features.data<T>();
    auto w = weight.data<T>();
    auto dw = d_weight.data<T>();

    dConvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);

    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}
