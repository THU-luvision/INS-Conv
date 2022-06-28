// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/core/Tensor.h>
template <typename T>
double dDeconvolution_forward2(T *inFeatures, T *outFeatures, T *w,
                               RuleBook _rules, Int input_nPlanes,
                               Int input_stride, Int output_nPlanes,
                               Int output_stride);

template <typename T>
void dDeconvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                 T *w, T *dw, RuleBook _rules,
                                 Int input_nPlanes, Int input_stride,
                                 Int output_nPlanes, Int output_stride);

template <typename T>
double dDeconvolution_incre_forward2(T *inFeatures, T *outFeatures, T *w,
			       RuleBook _rules, Int input_nPlanes,
			       Int input_stride, Int output_nPlanes,
			       Int output_stride, T * pre_input_feats, Int *pre_exist_input, Int *pre_exist_out);



template <typename T, Int Dimension>
double cuda_Deconvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias) {
EASY_FUNCTION(profiler::colors::Brick);
  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
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

    return dDeconvolution_forward2<T>(iF, oF, w, _rules, ip, ip, op, op);
  } else {
    return 0;
  }
}


template <typename T, Int Dimension>
double cuda_Incre_Deconvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias, Metadata<Dimension> &pre_m, at::Tensor pre_input_feats) {
    EASY_FUNCTION(profiler::colors::Brick);
    EASY_BLOCK("get rulebook");
  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  Int nActiveOut = m.getNActive(outputSize);
  Int nActiveIn = m.getNActive(inputSize);
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
    EASY_END_BLOCK;
    EASY_BLOCK("deconvolution forward");
    Int *pre_exist_input = m.getPreExistMaps(inputSize, pre_m);
    Int *pre_exist_out = m.getPreExistMaps(outputSize, pre_m);
    double flop= dDeconvolution_incre_forward2<T>(iF, oF, w, _rules, ip, ip, op, op, pre_input_feats.data<T>(), pre_exist_input, pre_exist_out);
  return flop;
  } else {
    return 0;
  }
}

template <typename T, Int Dimension>
void cuda_Deconvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor d_weight,
    /*cuda float*/ at::Tensor d_bias) {

  auto _rules =
      m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
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

    dDeconvolution_backward_dW2<T>(iF, diF, doF, w, dw, _rules, ip, ip, op, op);
    if (d_bias.numel()) {
      auto db = d_bias.data<T>();
      Convolution_bp_bias(doF, db, op, nActiveOut);
    }
  }
}
