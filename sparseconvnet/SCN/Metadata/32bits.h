// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>

// Using 32 bit integers for coordinates and memory calculations.

using Int = int32_t;

// Point<dimension> is a point in the d-dimensional integer lattice
// (i.e. square-grid/cubic-grid, ...)
template <Int dimension> using Point = std::array<Int, dimension>;

template <Int dimension>
Point<dimension> LongTensorToPoint(/*long*/ at::Tensor &t) {
  Point<dimension> p;
  long *td = t.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  return p;
}



#define DILATED_RATE_HASH 600000
template <Int dimension>
Point<2 * dimension> TwoLongTensorsToPoint_Dilation(/*long*/ at::Tensor &t0,
                                           /*long*/ at::Tensor &t1,
                                           int dilated_rate) {
  //dilated convolution equals to sparse convolution when dilated_rate = 1;
  dilated_rate = (dilated_rate - 1) * DILATED_RATE_HASH;
  Point<2 * dimension> p;
  long *td;
  td = t0.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  td = t1.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + dimension] = td[i] + dilated_rate;
  return p;
}


template <Int dimension>
Point<2 * dimension> TwoLongTensorsToPoint(/*long*/ at::Tensor &t0,
                                           /*long*/ at::Tensor &t1) {
  Point<2 * dimension> p;
  long *td;
  td = t0.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  td = t1.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + dimension] = td[i];
  return p;
}
template <Int dimension>
Point<3 * dimension> ThreeLongTensorsToPoint(/*long*/ at::Tensor &t0,
                                             /*long*/ at::Tensor &t1,
                                             /*long*/ at::Tensor &t2) {
  Point<3 * dimension> p;
  long *td;
  td = t0.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  td = t1.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + dimension] = td[i];
  td = t2.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + 2 * dimension] = td[i];
  return p;
}

// FNV Hash function for Point<dimension>
template <Int dimension> struct IntArrayHash {
  std::size_t operator()(Point<dimension> const &p) const {
    Int hash = 16777619;
    for (auto x : p) {
      hash *= 2166136261;
      hash ^= x;
    }
    return hash;
  }
};




#define at_kINT at::kInt
