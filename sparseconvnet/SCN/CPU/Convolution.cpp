// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstring>
template <typename T>
at::Tensor rule_index_select(at::Tensor src, Int nRules,
                       Int *rules) {
  auto n = src.size(1);
  auto target = at::empty({nRules, n}, src.type());
  auto t_ptr = target.data<T>();
  auto s_ptr = src.data<T>();
  #pragma omp parallel for
  for (Int i = 0; i < nRules; ++i)
    std::memcpy(t_ptr + i * n, s_ptr + rules[2 * i] * n, sizeof(T) * n);
  return target;
}
template <typename T>
void rule_index_add_(at::Tensor target, at::Tensor src, Int nRules,
                     Int *rules) {
  auto t_ptr = target.data<T>();
  auto s_ptr = src.data<T>();
  auto n = target.size(1);
  #pragma omp parallel for
  for (Int i = 0; i < nRules; ++i) {
    auto t = t_ptr + rules[2 * i] * n;
    auto s = s_ptr + i * n;
    for (int j = 0; j < n; ++j)
      t[j] += s[j];
  }
}

template <typename T, Int Dimension>
double cpu_Convolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor bias) {
  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op;
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto w = weight.select(0, i);
      // auto output_rows = at::mm(input_rows, w);
      // output_features.index_add_(0, rt.select(1, 1), output_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto w = weight.select(0, i);
      auto output_rows = at::mm(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1]);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_Convolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor d_weight, /*float*/ at::Tensor d_bias) {

  auto _rules =
      m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto d_output_rows = d_output_features.index_select(0, rt.select(1,
      // 1));
      // at::mm_out(dw, input_rows.t(), d_output_rows);
      // auto d_input_rows = at::mm(d_output_rows, w.t());
      // d_input_features.index_add_(0, rt.select(1, 0), d_input_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto d_output_rows = rule_index_select<T>(d_output_features, nRules, &r[1]);
      at::mm_out(dw, input_rows.t(), d_output_rows);
      auto d_input_rows = at::mm(d_output_rows, w.t());
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0]);
    }
  }
}

template <typename T, Int Dimension>
double cpu_SubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor weight,
    /*float*/ at::Tensor bias) {
  auto _rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true);
  Int nActive = m.getNActive(inputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op;
      // auto  rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto w = weight.select(0, i);
      // auto output_rows = at::mm(input_rows, w);
      // output_features.index_add_(0, rt.select(1, 1), output_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto w = weight.select(0, i);
      auto output_rows = at::mm(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1]);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_SubmanifoldConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor filterSize,
    Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor d_weight,
    /*float*/ at::Tensor d_bias,
    int dilated_rate) {

  auto _rules = m.getSubmanifoldRuleBook(inputSize, filterSize, true,dilated_rate);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto d_output_rows = d_output_features.index_select(0, rt.select(1,
      // 1));
      // at::mm_out(dw, input_rows.t(), d_output_rows);
      // auto d_input_rows = at::mm(d_output_rows, w.t());
      // d_input_features.index_add_(0, rt.select(1, 0), d_input_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto d_output_rows = rule_index_select<T>(d_output_features, nRules, &r[1]);
      at::mm_out(dw, input_rows.t(), d_output_rows);
      auto d_input_rows = at::mm(d_output_rows, w.t());
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0]);
    }
  }
}

template <typename T, Int Dimension>
double cpu_PermutohedralSubmanifoldConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor weight,
    /*float*/ at::Tensor bias) {
  auto _rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op;
      // auto  rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto w = weight.select(0, i);
      // auto output_rows = at::mm(input_rows, w);
      // output_features.index_add_(0, rt.select(1, 1), output_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto w = weight.select(0, i);
      auto output_rows = at::mm(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1]);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_PermutohedralSubmanifoldConvolution_backward(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor d_weight,
    /*float*/ at::Tensor d_bias) {

  auto _rules = m.getPermutohedralSubmanifoldRuleBook(inputSize, true);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto d_output_rows = d_output_features.index_select(0, rt.select(1,
      // 1));
      // at::mm_out(dw, input_rows.t(), d_output_rows);
      // auto d_input_rows = at::mm(d_output_rows, w.t());
      // d_input_features.index_add_(0, rt.select(1, 0), d_input_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto d_output_rows = rule_index_select<T>(d_output_features, nRules, &r[1]);
      at::mm_out(dw, input_rows.t(), d_output_rows);
      auto d_input_rows = at::mm(d_output_rows, w.t());
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0]);
    }
  }
}

template <typename T, Int Dimension>
double cpu_FullConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor weight,
    /*float*/ at::Tensor bias) {
  auto _rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActive = mOut.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op;
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto w = weight.select(0, i);
      // auto output_rows = at::mm(input_rows, w);
      // output_features.index_add_(0, rt.select(1, 1), output_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto w = weight.select(0, i);
      auto output_rows = at::mm(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1]);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_FullConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &mIn,
    Metadata<Dimension> &mOut,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor d_weight,
    /*float*/ at::Tensor d_bias) {

  auto _rules = mIn.getFullConvolutionRuleBook(inputSize, outputSize,
                                               filterSize, filterStride, mOut);
  Int nActive = mOut.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto d_output_rows = d_output_features.index_select(0, rt.select(1,
      // 1));
      // at::mm_out(dw, input_rows.t(), d_output_rows);
      // auto d_input_rows = at::mm(d_output_rows, w.t());
      // d_input_features.index_add_(0, rt.select(1, 0), d_input_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto d_output_rows = rule_index_select<T>(d_output_features, nRules, &r[1]);
      at::mm_out(dw, input_rows.t(), d_output_rows);
      auto d_input_rows = at::mm(d_output_rows, w.t());
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0]);
    }
  }
}

template <typename T, Int Dimension>
double cpu_RandomizedStrideConvolution_updateOutput(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor bias) {
  auto _rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, weight.size(2)});
  if (bias.numel() and nActive)
    output_features.copy_(bias);
  else
    output_features.zero_();

  double flops = 0;
  auto ip = weight.size(1);
  auto op = weight.size(2);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      flops += nRules * ip * op;
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto w = weight.select(0, i);
      // auto output_rows = at::mm(input_rows, w);
      // output_features.index_add_(0, rt.select(1, 1), output_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto w = weight.select(0, i);
      auto output_rows = at::mm(input_rows, w);
      rule_index_add_<T>(output_features, output_rows, nRules, &r[1]);
    }
  }
  return flops;
}

template <typename T, Int Dimension>
void cpu_RandomizedStrideConvolution_backward(
    /*long*/ at::Tensor inputSize, /*long*/ at::Tensor outputSize,
    /*long*/ at::Tensor filterSize,
    /*long*/ at::Tensor filterStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor weight,
    /*float*/ at::Tensor d_weight, /*float*/ at::Tensor d_bias) {

  auto _rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, filterSize,
                                              filterStride, true);
  Int nActive = m.getNActive(inputSize);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (nActive and d_bias.numel())
    at::sum_out(d_bias, d_output_features, {0}, false);
  for (Int i = 0; i < (Int)_rules.size(); i++) {
    auto r = _rules[i];
    int nRules = r.size() / 2;
    if (nRules) {
      auto w = weight.select(0, i);
      auto dw = d_weight.select(0, i);
      // auto rt = torch::CPU(at_kINT).tensorFromBlob(&r[0], {nRules, 2});
      // auto input_rows = input_features.index_select(0, rt.select(1, 0));
      // auto d_output_rows = d_output_features.index_select(0, rt.select(1,
      // 1));
      // at::mm_out(dw, input_rows.t(), d_output_rows);
      // auto d_input_rows = at::mm(d_output_rows, w.t());
      // d_input_features.index_add_(0, rt.select(1, 0), d_input_rows);
      auto input_rows = rule_index_select<T>(input_features, nRules, &r[0]);
      auto d_output_rows = rule_index_select<T>(d_output_features, nRules, &r[1]);
      at::mm_out(dw, input_rows.t(), d_output_rows);
      auto d_input_rows = at::mm(d_output_rows, w.t());
      rule_index_add_<T>(d_input_features, d_input_rows, nRules, &r[0]);
    }
  }
}
