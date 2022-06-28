// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "../Metadata/Metadata.h"
#include <ATen/core/Formatting.h>
#include <ATen/core/Tensor.h>
#include <easy/arbitrary_value.h>


template <typename T>
void bn_f(T *iF, T *oF, Int nPlanes, Int input_stride, Int output_stride,
          Int nActive, T *saveMean, T *saveInvStd, T *runningMean,
          T *runningVar, T *weight, T *bias, T eps, T momentum, bool train,
          T leakiness);

template <typename T>
void bn_b(T *input_features, T *d_input_features, T *output_features,
          T *d_output_features, Int nPlanes, Int input_stride,
          Int output_stride, Int nActive, T *saveMean, T *saveInvStd,
          T *runningMean, T *runningVar, T *weight, T *bias, T *d_weight,
          T *d_bias, T leakiness);

template <typename T>
void inc_bn_f(T *iF, T *oF, Int nPlanes, Int input_stride, Int output_stride,
          Int nActive, T *saveMean, T *saveInvStd, T *runningMean,
          T *runningVar, T *weight, T *bias, T eps, T momentum, bool train,
          T leakiness, Int *pre_exist, T *pre_output_feats, T *pre_input_feats);

template <typename T>
void inc_bn_b(T *input_features, T *d_input_features, T *output_features,
          T *d_output_features, Int nPlanes, Int input_stride,
          Int output_stride, Int nActive, T *saveMean, T *saveInvStd,
          T *runningMean, T *runningVar, T *weight, T *bias, T *d_weight,
          T *d_bias, T leakiness, Int *pre_exist, T *pre_output_feats, T *pre_input_feats);


template <typename T>
void cuda_BatchNormalization_updateOutput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor saveMean,
    /*cuda float*/ at::Tensor saveInvStd, /*cuda float*/ at::Tensor runningMean,
    /*cuda float*/ at::Tensor runningVar,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor bias, T eps,
    T momentum, bool train, T leakiness) {
  EASY_FUNCTION(profiler::colors::Blue200);
  output_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    bn_f(input_features.data<T>(), output_features.data<T>(), nPlanes,
         input_stride, output_stride, nActive, saveMean.data<T>(),
         saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),
         OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps,
         momentum, train, leakiness);
  }
}

template <typename T, Int Dimension>
void cuda_Incre_BatchNormalization_updateOutput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor saveMean,
    /*cuda float*/ at::Tensor saveInvStd, /*cuda float*/ at::Tensor runningMean,
    /*cuda float*/ at::Tensor runningVar,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor bias, T eps,
    T momentum, bool train, T leakiness, Metadata<Dimension> &m, Metadata<Dimension> &pre_m, 
    at::Tensor pre_output_feats, at::Tensor pre_input_feats, at::Tensor inputSize) {
  EASY_FUNCTION(profiler::colors::Blue100);
  output_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);

    Point<Dimension> spatial_size = LongTensorToPoint<Dimension>(inputSize);
    Int *out_points = m.GPU_grids[spatial_size][0].pHash->getAllPoints();
    Int *pre_exist = m.getPreExistMaps(inputSize, pre_m); 

    inc_bn_f(input_features.data<T>(), output_features.data<T>(), nPlanes,
         input_stride, output_stride, nActive, saveMean.data<T>(),
         saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),
         OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps,
         momentum, train, leakiness, pre_exist, pre_output_feats.data<T>(), pre_input_feats.data<T>());
  }
}


template <typename T>
void cuda_BatchNormalization_backward(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor saveMean, /*cuda float*/ at::Tensor saveInvStd,
    /*cuda float*/ at::Tensor runningMean,
    /*cuda float*/ at::Tensor runningVar, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias,
    /*cuda float*/ at::Tensor d_weight, /*cuda float*/ at::Tensor d_bias,
    T leakiness) {

  d_input_features.resize_as_(d_output_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    bn_b(input_features.data<T>(), d_input_features.data<T>(),
         output_features.data<T>(), d_output_features.data<T>(), nPlanes,
         input_stride, output_stride, nActive, saveMean.data<T>(),
         saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),
         OptionalTensorData<T>(weight), OptionalTensorData<T>(bias),
         OptionalTensorData<T>(d_weight), OptionalTensorData<T>(d_bias),
         leakiness);
  }
}
