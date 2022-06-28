// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CONVOLUTIONRULES_H
#define CONVOLUTIONRULES_H
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "RectangularRegions.h"
#include "Metadata.h"
#include <algorithm>

// only supports 3D convolutions now, and will be modified in the future maybe.
// coordinate exchange
template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg(SparseGrid<dimension> &inputGrid,
                                           SparseGrid<dimension> &outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize,
                                           const std::vector<Float3> &input_normal,
                                           std::vector<Float3> &output_normal) {
  EASY_FUNCTION(profiler::colors::Green100); 
//right hand coordinate,
  int index[6*8] = {0,1,2,3,4,5,6,7,
                    6,7,4,5,2,3,0,1,
                    2,3,6,7,0,1,4,5,
                    4,5,0,1,6,7,2,3,
                    1,5,3,7,0,4,2,6,
                    4,0,6,2,5,1,7,3};
  Int outputStart = outputGrid.ctr;
  std::vector<Int> inputObservations = std::vector<Int>(inputGrid.mp.size());

 // orientation is determined by the first one?


  RuleBook candidate_rules = std::vector<std::vector<Int>>(volume<dimension>(size));
  for(int i = 0; i < volume<dimension>(size);i++)
  {
    candidate_rules[i].reserve(inputGrid.mp.size());
  }
  for (auto const &inIter : inputGrid.mp) {
    auto outRegion = OutputRegionCalculator<dimension>(
        inIter.first, size, stride, outputSpatialSize);
//    printf("new output\r\n\r\n");
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(inIter.first);
      auto outIter = outputGrid.mp.find(j);
      if (outIter == outputGrid.mp.end()) {
        outIter =
            outputGrid.mp.insert(std::make_pair(j, outputGrid.ctr++)).first;
        inputObservations[outIter->second - outputStart ] = 0;
        output_normal.push_back(Float3(0,0,0));
      }
      inputObservations[outIter->second - outputStart ]++;
      output_normal[outIter->second] += input_normal[inIter.second+inputGrid.ctr]; // find mapping based on
//      printf("output point: %d %d %d %d %d\r\n",j[0],j[1],j[2], rulesOffset, outIter->second);
      candidate_rules[rulesOffset].push_back(inIter.second + inputGrid.ctr);
      candidate_rules[rulesOffset].push_back(outIter->second);
    }
  }

  std::vector<Int> oriented_index = std::vector<Int>(outputGrid.ctr - outputStart );
  for(int i = outputStart ; i < outputGrid.ctr; i++)
  {
    output_normal[i] /= inputObservations[i - outputStart ];
    output_normal[i].normalize();
    oriented_index[i - outputStart ] = OrientedFilter(output_normal[i]);
  }
  rules.resize(volume<dimension>(size));
  // there might be some problems here
  // target: low level texture information && high level geometry information
  // minimize the rotation invariance
  for(int i = 0; i < candidate_rules.size();i++)
  {
    for(int k = 0; k < candidate_rules[i].size();k+=2)
    {
        int ori_index = index[oriented_index[candidate_rules[i][k+1] - outputStart ] * 8 + i];
 #if 0
        rules[i].push_back(candidate_rules[i][k]);
        rules[i].push_back(candidate_rules[i][k+1]);
 #else
        rules[ori_index].push_back(candidate_rules[i][k]);
        rules[ori_index].push_back(candidate_rules[i][k+1]);
 #endif
    }
  }
}


template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg(SparseGrid<dimension> &inputGrid,
                                           SparseGrid<dimension> &outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize) {
  rules.resize(volume<dimension>(size));

  for (auto const &inIter : inputGrid.mp) {
    auto outRegion = OutputRegionCalculator<dimension>(
        inIter.first, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(inIter.first);
      auto outIter = outputGrid.mp.find(j);
      if (outIter == outputGrid.mp.end()) {
        outIter =
            outputGrid.mp.insert(std::make_pair(j, outputGrid.ctr++)).first;
      }
      rules[rulesOffset].push_back(inIter.second + inputGrid.ctr);
      rules[rulesOffset].push_back(outIter->second);
    }
  }
}

// GPU
#ifdef GPU_GRID



// only supports 3D convolutions now, and will be modified in the future maybe.
// coordinate exchange
// maybe 3x3 is a better choice, but just a little slower, need to check in the future

void dGenerateSpatialNewPoint (Int* d_prev_all_point,             // size = num_active_point * dim
	Int* d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
	long* d_size,
    long* d_stride,
    long* d_output_spatial_size,
    Int num_active_point,
    long maxi_sizec,
	Int ndim);


template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg(GPU_SparseGrid<dimension> &gpu_inputGrid,
                                           GPU_SparseGrid<dimension> &gpu_outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize,
                                           const std::vector<Float3> &input_normal,
                                           std::vector<Float3> &output_normal) {
  EASY_FUNCTION(profiler::colors::Green50);
  EASY_BLOCK("gen_outpoint");
  Int nActiveInput = gpu_inputGrid.pHash->getCompactingSize();

  rules.resize(volume<dimension>(size));

  Int* in_points_flat = new Int[nActiveInput * dimension];  // size = nActiveInput * dimension
  gpuErrchk(cudaMemcpy(in_points_flat,
                       gpu_inputGrid.pHash->getAllPoints(),
                       sizeof(Int) * nActiveInput * dimension,
                       cudaMemcpyDeviceToHost));

  Point<dimension> p;
  Points<dimension> out_points;
  Ints out_index;
  EASY_BLOCK("transformat");
  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      out_points.push_back(j);
    }
  }

  EASY_END_BLOCK;
  EASY_END_BLOCK;
  EASY_BLOCK("insert retrieve");
  gpu_outputGrid.pHash->insert_points(out_points);
  gpu_outputGrid.pHash->retrieve_points(out_points, out_index);

  EASY_END_BLOCK;
  EASY_BLOCK("Normal");
  Int query_index = 0;

  //right hand coordinate,
  int index[6*8] = {0,1,2,3,4,5,6,7,
                    6,7,4,5,2,3,0,1,
                    2,3,6,7,0,1,4,5,
                    4,5,0,1,6,7,2,3,
                    1,5,3,7,0,4,2,6,
                    4,0,6,2,5,1,7,3};
  std::vector<Int> inputObservations = std::vector<Int>(gpu_outputGrid.pHash->size);
  output_normal.resize(gpu_outputGrid.ctr + gpu_outputGrid.pHash->size);

  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      inputObservations[out_index[query_index]]++;
      output_normal[gpu_outputGrid.ctr + out_index[query_index]] += input_normal[i + gpu_inputGrid.ctr];
      query_index+=1;
    }
  }
  for(int k = 0; k < gpu_outputGrid.pHash->size;k++)
  {
    if(inputObservations[k] > 0)
    {
        output_normal[gpu_outputGrid.ctr + k] /= inputObservations[k];
    }
  }
  EASY_END_BLOCK;
  EASY_BLOCK("query");
  query_index = 0;
  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(p);
      Int oriIndex = OrientedFilter(output_normal[gpu_outputGrid.ctr + out_index[query_index++]]);
      Int newRuleOffset = index[oriIndex * 8 + rulesOffset];
      rules[newRuleOffset].push_back(i + gpu_inputGrid.ctr);
      rules[newRuleOffset].push_back(gpu_outputGrid.ctr + out_index[query_index++]);
    }
  }

  gpu_outputGrid.ctr += gpu_outputGrid.pHash->size;

  delete[] in_points_flat;
EASY_END_BLOCK;
}
// utils to debug
template <Int dimension>
bool point_cmp(Point<dimension> a,Point<dimension> b)
{
  for(Int i=0;i<dimension;i++)
  {
    if(a[i]<b[i])return true;
    if(a[i]>b[i])return false;
  }
  return false;
}
template <Int dimension>
void point_sort(vector<Point<dimension>> &a)
{
  sort(a.begin(),a.end(),point_cmp<dimension>);
}


// only support when size == stride case
template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg(GPU_SparseGrid<dimension> &gpu_inputGrid,
                                           GPU_SparseGrid<dimension> &gpu_outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize) {

  // Note that this is a special case for downscaling
  EASY_FUNCTION(profiler::colors::Green100);
  for(int i = 0; i < dimension; i++)
  {
//      printf("convolution kernel: %d %d %d\r\n", i,size[i], stride[i]);
      assert(size[i] == 2);
      assert(stride[i] == 2);
  }
  rules.resize(volume<dimension>(size));
  Int nActiveInput = gpu_inputGrid.pHash->getCompactingSize();

  Int* in_points_flat = new Int[nActiveInput * dimension];  // size = nActiveInput * dimension
  gpuErrchk(cudaMemcpy(in_points_flat,
                       gpu_inputGrid.pHash->getAllPoints(),
                       sizeof(Int) * nActiveInput * dimension,
                       cudaMemcpyDeviceToHost));
  // rewrite the rules generation part to get faster implementation
  Point<dimension> p;
  Points<dimension> out_points;
  Ints out_index;

  // note that this is a mapping from in_points to out_points


  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      out_points.push_back(j);
    }
  }

  gpu_outputGrid.pHash->insert_points(out_points);
  gpu_outputGrid.pHash->retrieve_points(out_points, out_index);

  Int query_index = 0;
  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(p);
 //     printf("%d %d %d %d %d %d %d\r\n", p[0],p[1],p[2],j[0],j[1],j[2],rulesOffset);
      rules[rulesOffset].push_back(i + gpu_inputGrid.ctr);
      rules[rulesOffset].push_back(gpu_outputGrid.ctr + out_index[query_index++]);
    }
  }

  gpu_outputGrid.ctr += gpu_outputGrid.pHash->size;

  delete[] in_points_flat;
}

void d_Convolution_GenerateOutputRules(uint32_t * d_in_points, uint32_t * d_output_points, uint32_t * d_output_index,
                                       RuleBook &rules,Int num,  Int dimension, Int filterSize, Int input_offset);

at::Tensor FlatPoints(const at::Tensor &input_points);

template <Int dimension>
at::Tensor ResolutionBasedScatteringCuda(at::Tensor &points_lr, at::Tensor &points_hr, Int stride)
{
    assert(dimension == points_lr.size(1));
    int lr_point_num = points_lr.size(0);
    int hr_point_num = points_hr.size(0);
    GPU_SparseGrid<dimension> gpu_grid;
    at::Tensor point_lr_flat = FlatPoints(points_lr);
    at::Tensor point_hr_flat = FlatPoints(points_hr);
    at::Tensor point_hr_query = point_hr_flat / stride;
    at::Tensor hr2lr = torch::empty({hr_point_num}, at::CUDA(at_kINT));
    gpu_grid.pHash->insert((uint32_t* )point_lr_flat.data<Int>(), lr_point_num);
    gpu_grid.pHash->retrieve((uint32_t* )point_hr_query.data<Int>(), (uint32_t* )hr2lr.data<Int>(),hr_point_num);

    return hr2lr;
}

template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg_FastDownSampleMode(GPU_SparseGrid<dimension> &gpu_inputGrid,
                                           GPU_SparseGrid<dimension> &gpu_outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize) {

  // Note that this is a special case for downscaling
  EASY_FUNCTION(profiler::colors::Green100);
  for(int i = 0; i < dimension; i++)
  {
//      printf("convolution kernel: %d %d %d\r\n", i,size[i], stride[i]);
      assert(size[i] == 2);
      assert(stride[i] == 2);
  }

  clock_t start,end;
  // rule resized to kernel volumn size.
  rules.resize(volume<dimension>(size));
  Int nActiveInput = gpu_inputGrid.pHash->getCompactingSize();

  at::Tensor d_in_points = at::empty({nActiveInput * dimension}, at::CUDA(at_kINT));
  gpuErrchk(cudaMemcpy(d_in_points.data<Int>(),
                       gpu_inputGrid.pHash->getAllPoints(),
                       sizeof(Int) * nActiveInput * dimension,
                       cudaMemcpyDeviceToDevice));
  // use grid down sample. if points coord too small, there might be some problem.
  at::Tensor d_out_points = d_in_points / 2;
  at::Tensor d_results = at::empty({nActiveInput}, at::CUDA(at_kINT));
  gpu_outputGrid.pHash->insert((uint32_t* )d_out_points.data<Int>(), nActiveInput);
  // get unique point index
  gpu_outputGrid.pHash->retrieve((uint32_t* )d_out_points.data<Int>(), (uint32_t* )d_results.data<Int>(),nActiveInput);
  d_results += gpu_outputGrid.ctr;
  d_Convolution_GenerateOutputRules((uint32_t* )d_in_points.data<Int>(),(uint32_t* )d_out_points.data<Int>(),
                                    (uint32_t* )d_results.data<Int>(),
                                    rules, nActiveInput, dimension, 2,gpu_inputGrid.ctr);
  gpu_outputGrid.ctr += gpu_outputGrid.pHash->size;
#if 0
  Int* in_points_flat = new Int[nActiveInput * dimension];  // size = nActiveInput * dimension
  gpuErrchk(cudaMemcpy(in_points_flat,
                       gpu_inputGrid.pHash->getAllPoints(),
                       sizeof(Int) * nActiveInput * dimension,
                       cudaMemcpyDeviceToHost));

  // generate output points

  // rewrite the rules generation part to get faster implementation
  Point<dimension> p;
  Points<dimension> out_points;
  Ints out_index;

  // note that this is a mapping from in_points to out_points


  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      out_points.push_back(j);
    }
  }

  gpu_outputGrid.pHash->insert_points(out_points);
  gpu_outputGrid.pHash->retrieve_points(out_points, out_index);

  Int query_index = 0;
  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(p);
 //     printf("%d %d %d %d %d %d %d\r\n", p[0],p[1],p[2],j[0],j[1],j[2],rulesOffset);
      rules[rulesOffset].push_back(i + gpu_inputGrid.ctr);
      rules[rulesOffset].push_back(gpu_outputGrid.ctr + out_index[query_index++]);
    }
  }
  delete[] in_points_flat;
#endif
}
#if 0
template <Int dimension>
void Convolution_InputSgToRulesAndOutputSg(GPU_SparseGrid<dimension> &gpu_inputGrid,
                                           GPU_SparseGrid<dimension> &gpu_outputGrid,
                                           RuleBook &rules, long *size,
                                           long *stride, long *inputSpatialSize,
                                           long *outputSpatialSize) {

  EASY_FUNCTION(profiler::colors::Green100);
  EASY_BLOCK("GPU to CPU");
  rules.resize(volume<dimension>(size));
  Int nActiveInput = gpu_inputGrid.pHash->getCompactingSize();

  Int* in_points_flat = new Int[nActiveInput * dimension];  // size = nActiveInput * dimension
  gpuErrchk(cudaMemcpy(in_points_flat, 
                       gpu_inputGrid.pHash->getAllPoints(), 
                       sizeof(Int) * nActiveInput * dimension,
                       cudaMemcpyDeviceToHost));
  
  Point<dimension> p;
  Points<dimension> out_points;
  EASY_END_BLOCK;
  // EASY_BLOCK("transformat");
  // for (Int i = 0; i < nActiveInput; i++)  {
  //   for(Int k = 0; k < dimension; k++)  {
  //     p[k] = in_points_flat[i + k * nActiveInput];
  //   }
  //   auto outRegion = OutputRegionCalculator<dimension>(
  //       p, size, stride, outputSpatialSize);
  //   for (auto j : outRegion) {
  //     out_points.push_back(j);
  //   }
  // }
    
  #ifdef NEW_SPTIAL_POINT
    EASY_BLOCK("new gen output");
    // yet a more efficient way
    Int *d_prev_all_point=NULL;
    Int *d_next_all_point=NULL;
    long *d_size=NULL;
    long *d_stride=NULL;
    long *d_outputSpatialSize=NULL;
    d_prev_all_point=gpu_inputGrid.pHash->getAllPoints();
    // small mem allocate 
    gpuErrchk(cudaMalloc((void **)&d_size, sizeof(long) * dimension));
    gpuErrchk(cudaMemcpy(d_size,
                        size,
                        sizeof(long) * dimension,
                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **)&d_stride, sizeof(long) * dimension));
    gpuErrchk(cudaMemcpy(d_stride,
                        stride,
                        sizeof(long) * dimension,
                        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **)&d_outputSpatialSize, sizeof(long) * dimension));
    gpuErrchk(cudaMemcpy(d_outputSpatialSize,
                        outputSpatialSize,
                        sizeof(long) * dimension,
                        cudaMemcpyHostToDevice));
    // identify the output size
    long maxi_sizec=1;
    // Point<dimension> lb, ub,;
    for (Int i = 0; i < dimension; i++) {
      // lb[i] = std::max(0L, (input[i] - size[i] + stride[i]) / stride[i]);
      // ub[i] = std::min(outputSpatialSize[i] - 1, input[i] / stride[i]);
      maxi_sizec*= ( size[i] - stride[i]) / stride[i]+1;
    }
    gpuErrchk(cudaMalloc((void **)&d_next_all_point, sizeof(Int) * nActiveInput * maxi_sizec * dimension));
    gpuErrchk(cudaMemset(d_next_all_point, 0, sizeof(Int) * nActiveInput * maxi_sizec * dimension));

      dGenerateSpatialNewPoint(
        d_prev_all_point,             // size = num_active_point * dim
        d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
        d_size,
        d_stride,
        d_outputSpatialSize,
        nActiveInput,
        maxi_sizec,
        dimension);
      
      gpuErrchk( cudaDeviceSynchronize() );
      EASY_END_BLOCK;
      /*
        EASY_BLOCK("GPU to CPU");
        Int* tmp_outpoint=new Int[nActiveInput * maxi_sizec * dimension];
        gpuErrchk(cudaMemcpy(tmp_outpoint,
                            d_next_all_point,
                            sizeof(Int) * nActiveInput * maxi_sizec * dimension,
                            cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_next_all_point));
        for (Int i = 0; i < nActiveInput; i++)  {
          for (Int j = 0; j < maxi_sizec; j++)
          {
            for(Int k = 0; k < dimension; k++)  
            {
              p[k] = tmp_outpoint[(i + k * nActiveInput)*maxi_sizec+j];
            }
            out_points.push_back(p);
          }
        }
        delete tmp_outpoint;
          // can be more efficient here by modify porting issue:
      */
  

/*
  Int *d_prev_all_point=NULL;
  Int *d_next_all_point=NULL;
  long *d_size=NULL;
  long *d_stride=NULL;
  long *d_outputSpatialSize=NULL;
  gpuErrchk(cudaMalloc((void **)&d_prev_all_point, sizeof(Int) * nActiveInput * dimension));
  gpuErrchk(cudaMemcpy(d_prev_all_point,
                      in_points_flat,
                      sizeof(Int) * nActiveInput * dimension,
                      cudaMemcpyHostToDevice));
  
  gpuErrchk(cudaMalloc((void **)&d_size, sizeof(long) * dimension));
  gpuErrchk(cudaMemcpy(d_size,
                      size,
                      sizeof(long) * dimension,
                      cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void **)&d_stride, sizeof(long) * dimension));
  gpuErrchk(cudaMemcpy(d_stride,
                      stride,
                      sizeof(long) * dimension,
                      cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void **)&d_outputSpatialSize, sizeof(long) * dimension));
  gpuErrchk(cudaMemcpy(d_outputSpatialSize,
                      outputSpatialSize,
                      sizeof(long) * dimension,
                      cudaMemcpyHostToDevice));

  // output point mem size(upper bound) would be inactive* maxi_sizec
  long all_stride=1,maxi_sizec=1;
  for(Int k = 0; k < dimension; k++)  {
      all_stride*=stride[k];
    }
  // Point<dimension> lb, ub,;
  for (Int i = 0; i < dimension; i++) {
    // lb[i] = std::max(0L, (input[i] - size[i] + stride[i]) / stride[i]);
    // ub[i] = std::min(outputSpatialSize[i] - 1, input[i] / stride[i]);
    maxi_sizec*= ( size[i] - stride[i]) / stride[i]+1;
  }
  #ifdef PRINT_NEW_SPTIAL_POINT
  printf("maxi_sizec:%d",maxi_sizec);
  #endif
  gpuErrchk(cudaMalloc((void **)&d_next_all_point, sizeof(Int) * nActiveInput * maxi_sizec * dimension));
  gpuErrchk(cudaMemset(d_next_all_point, 0, sizeof(Int) * nActiveInput * maxi_sizec * dimension));
  EASY_END_BLOCK;
  EASY_BLOCK("new gen output");
  dGenerateSpatialNewPoint(
    d_prev_all_point,             // size = num_active_point * dim
    d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
    d_size,
    d_stride,
    d_outputSpatialSize,
    nActiveInput,
    maxi_sizec,
    dimension);
  EASY_END_BLOCK;
  EASY_BLOCK("two ways portting");
  gpuErrchk( cudaDeviceSynchronize() );
  // FOR DEBUG
  Int* debug_outpoint=new Int[nActiveInput * maxi_sizec * dimension];
  gpuErrchk(cudaMemcpy(debug_outpoint,
                      d_next_all_point,
                      sizeof(Int) * nActiveInput * maxi_sizec * dimension,
                      cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_prev_all_point));
  gpuErrchk(cudaFree(d_next_all_point));
  vector<Point<dimension>> debug_outputpoint;
  // Point<dimension> p;
    for (Int i = 0; i < nActiveInput; i++)  {
      for (Int j = 0; j < maxi_sizec; j++)
      {
        for(Int k = 0; k < dimension; k++)  
        {
          p[k] = debug_outpoint[(i + k * nActiveInput)*maxi_sizec+j];
        }
        debug_outputpoint.push_back(p);
      }
    }
  #ifdef PRINT_NEW_SPTIAL_POINT
  printf("\n%d %d\n",debug_outputpoint.size(),out_points.size());
   #endif
  point_sort<dimension>(debug_outputpoint);
  point_sort<dimension>(out_points);

  for(Int i=0;i<min(debug_outputpoint.size(),out_points.size());i++)
  {
    if(debug_outputpoint[i]!=out_points[i])
    {
      printf("two ways gen not same");
      break;
    }
     #ifdef PRINT_NEW_SPTIAL_POINT
    if(i+1==min(debug_outputpoint.size(),out_points.size()))
    {
      printf("two ways gen the same");
    }
    #endif
  }
  EASY_END_BLOCK;
  // cmp debug_outpoint and out_points
  // new transformate here
  // input gpumem all point() addr, outpoint addr, size, stride,outspsize,ndim, aactive
  EASY_END_BLOCK;
  EASY_END_BLOCK;

*/
#endif
  EASY_BLOCK("insert and retrieve");
  // gpu_outputGrid.pHash->insert_points(out_points);
  // gpu_outputGrid.pHash->retrieve_points(out_points, out_index);
  gpu_outputGrid.pHash->insert((uint32_t*)d_next_all_point,nActiveInput * maxi_sizec);
  uint32_t *d_results = NULL;  /* query results*/
  // Allocate memory for results 
  gpuErrchk(cudaMalloc((void**)&d_results, sizeof(uint32_t) * nActiveInput * maxi_sizec));
  gpu_outputGrid.pHash->retrieve((uint32_t*)d_next_all_point,d_results,nActiveInput * maxi_sizec);
  Ints out_index;
  out_index.resize(nActiveInput * maxi_sizec);
  gpuErrchk(cudaMemcpy(out_index.data(), d_results, sizeof(uint32_t) * nActiveInput * maxi_sizec, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_results));
  gpuErrchk( cudaDeviceSynchronize() );
  EASY_END_BLOCK;

  EASY_BLOCK("query");
  // can also be more efficient, but it's more complex because of the struct rulebook vector
  Int query_index = 0;
  for (Int i = 0; i < nActiveInput; i++)  {
    for(Int k = 0; k < dimension; k++)  {
      p[k] = in_points_flat[i + k * nActiveInput];
    }
    auto outRegion = OutputRegionCalculator<dimension>(
        p, size, stride, outputSpatialSize);
    for (auto j : outRegion) {
      auto inRegion = InputRegionCalculator<dimension>(j, size, stride);
      Int rulesOffset = inRegion.offset(p);
      rules[rulesOffset].push_back(i + gpu_inputGrid.ctr);
      rules[rulesOffset].push_back(gpu_outputGrid.ctr + out_index[query_index++]);
    }
  }

  gpu_outputGrid.ctr += gpu_outputGrid.pHash->size;
  
  delete[] in_points_flat;
  gpuErrchk(cudaFree(d_next_all_point));
  EASY_END_BLOCK;
}
#endif
#endif

template <Int dimension>
Int Convolution_InputSgsToRulesAndOutputSgs(
#ifdef GPU_GRID
                                            GPU_SparseGrids<dimension> &input_SGs,
                                            GPU_SparseGrids<dimension> &output_SGs,
#else
                                            SparseGrids<dimension> &input_SGs,
                                            SparseGrids<dimension> &output_SGs,
#endif
                                            RuleBook &rules, long *filterSize,
                                            long *filterStride,
                                            long *input_spatialSize,
                                            long *output_spatialSize, std::vector<Float3> &output_normal) {

  EASY_FUNCTION(profiler::colors::Green100);

  rules.clear();
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  Int output_nActive = 0;
  Int temp;
  for (Int i = 0; i < batchSize; i++) {
    auto &iSG = input_SGs[i];
    auto &oSG = output_SGs[i];
    oSG.ctr = output_nActive;



    Convolution_InputSgToRulesAndOutputSg<dimension>(
        iSG, oSG, rules, filterSize, filterStride, input_spatialSize,
        output_spatialSize);

    temp = output_nActive;
    output_nActive = oSG.ctr;
    oSG.ctr = temp;
  }
  // Debug: Print rulebook
#ifdef PRINT_CONVOLUTION
  printf("Convolution rules:\n");
  for (Int i = 0; i < (Int)rules.size(); i++) { 
    for (Int j = 0; j < (Int)rules[i].size(); j+=2)  {
      std::cout << "Offset: " << i << ", Rules: " << rules[i][j] << ", "<< rules[i][j+1] << std::endl; 
    }
    std::cout << std::endl; 
  } 
  printf("output_nActive = %d\n", output_nActive);
#endif

  return output_nActive;
}


#define DEBUG_FAST_CONV_RULES 0
template <Int dimension>
Int Convolution_InputSgsToRulesAndOutputSgs(
#ifdef GPU_GRID
                                            GPU_SparseGrids<dimension> &input_SGs,
                                            GPU_SparseGrids<dimension> &output_SGs,
#else
                                            SparseGrids<dimension> &input_SGs,
                                            SparseGrids<dimension> &output_SGs,
#endif
                                            RuleBook &rules, long *filterSize,
                                            long *filterStride,
                                            long *input_spatialSize,
                                            long *output_spatialSize,
                                            const std::vector<Float3> &input_normal,
                                            std::vector<Float3> &output_normal,
                                            int normal_guide_scale) {

  output_normal.clear();
  rules.clear();
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  // Int input_points = 0;
  Int output_nActive = 0;
  Int temp;
#if DEBUG_FAST_CONV_RULES
  RuleBook newRules;
  newRules.clear();
#endif
  for (Int i = 0; i < batchSize; i++) {
    auto &iSG = input_SGs[i];
    auto &oSG = output_SGs[i];
    oSG.ctr = output_nActive;
    if(input_normal.empty() || input_spatialSize[0] < normal_guide_scale)
    {
#if DEBUG_FAST_CONV_RULES
        cudaDeviceSynchronize();
        clock_t start,end;
        double time_gt,time_fast;
        start = clock();
#endif

        Convolution_InputSgToRulesAndOutputSg_FastDownSampleMode<dimension>(
            iSG, oSG, rules, filterSize, filterStride, input_spatialSize,
            output_spatialSize);
#if DEBUG_FAST_CONV_RULES

        end = clock();
        time_gt = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
        start = clock();

        GPU_SparseGrid<dimension> output_SG;
        output_SG.ctr = output_nActive;
        printf("input ctr: %d %d\r\n", iSG.ctr, output_SG.ctr);
        Convolution_InputSgToRulesAndOutputSg<dimension>(
            iSG, output_SG, newRules, filterSize, filterStride, input_spatialSize,
            output_spatialSize);
        cudaDeviceSynchronize();
        end = clock();
        time_fast = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
        printf("timing: %f %f\r\n", time_gt, time_fast);
        for(int i = 0; i < rules.size();i++)
        {
            printf("%d %d\r\n",newRules[i].size(), rules[i].size());
            for(int j = 0; j < rules[i].size();j+=2)
            {
                if(newRules[i][j] != rules[i][j] || newRules[i][j+1] != rules[i][j+1])
                {
                    printf("%d %d %d %d %d %d\r\n",i,j,newRules[i][j], newRules[i][j+1],rules[i][j],rules[i][j+1]);
                    exit(0);
                }
            }
        }
        printf("pass verification!\r\n");
#endif

    }
    else
    {
//        printf("normal guide scale : %d\r\n", input_spatialSize[0]);
        Convolution_InputSgToRulesAndOutputSg<dimension>(
        iSG, oSG, rules, filterSize, filterStride, input_spatialSize,
        output_spatialSize, input_normal, output_normal);
    }
    temp = output_nActive;
    output_nActive = oSG.ctr;
    oSG.ctr = temp;
  }

  // Debug: Print rulebook
#ifdef PRINT_CONVOLUTION
  printf("Convolution rules with normal:\n");
  for (Int i = 0; i < (Int)rules.size(); i++) { 
    for (Int j = 0; j < (Int)rules[i].size(); j+=2)  {
      std::cout << "Offset: " << i << ", Rules: " << rules[i][j] << ", "<< rules[i][j+1] << std::endl; 
    }
    std::cout << std::endl; 
  } 
  printf("output_nActive = %d\n", output_nActive);
#endif

  return output_nActive;
}

template <Int dimension>
Int Convolution_InputSgsToRulesAndOutputSgs_OMP(
    SparseGrids<dimension> &input_SGs, SparseGrids<dimension> &output_SGs,
    RuleBook &rules, long *filterSize, long *filterStride,
    long *input_spatialSize, long *output_spatialSize) {
  rules.clear();
  rules.resize(volume<dimension>(filterSize));
  output_SGs.clear();
  Int batchSize = input_SGs.size();
  output_SGs.resize(batchSize);
  std::vector<RuleBook> rbs(batchSize);
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < batchSize; i++)
      Convolution_InputSgToRulesAndOutputSg<dimension>(
          input_SGs[i], output_SGs[i], rbs[i], filterSize, filterStride,
          input_spatialSize, output_spatialSize);
  }
  Int output_nActive = 0;
  for (Int i = 0; i < batchSize; i++) {
    // Parallel assignment:
    // output_nActive     <-  output_nActive+output_SGs[i].ctr
    // output_SGs[i].ctr  <-  output_nActive
    Int tmp = output_nActive;
    output_nActive += output_SGs[i].ctr;
    output_SGs[i].ctr = tmp;
  }
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < (Int)rules.size(); i++) {
      auto &R = rules[i];
      for (Int j = 0; j < batchSize; j++) {
        auto &r = rbs[j][i];
        auto offset = output_SGs[j].ctr;
        for (Int k = 0; k < (Int)r.size();) {
          R.push_back(r[k++]);
          R.push_back(r[k++] + offset);
        }
      }
    }
  }
  return output_nActive;
}

// for each active site, list of (inputFeatureNumber,batchIdx, spatialOffset)
// triples
template <Int dimension>
void SparseToDense_InputSgsToRulesAndOutputSgs(
    SparseGrids<dimension> &input_SGs, RuleBook &rules, long *spatialSize) {
  Int batchSize = input_SGs.size();
  rules.clear();
  rules.resize(batchSize);
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; ++i) {
    lb[i] = 0;
    ub[i] = spatialSize[i] - 1;
  }
  auto region = RectangularRegion<dimension>(lb, ub);
  for (Int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    auto &iSG = input_SGs[batchIdx];
    for (auto const &inIter : iSG.mp) {
      rules[batchIdx].push_back(inIter.second + iSG.ctr);
      rules[batchIdx].push_back(region.offset(inIter.first));
    }
  }
}

template <Int dimension>
void SparseToDense_InputSgsToRulesAndOutputSgs_OMP(
    SparseGrids<dimension> &input_SGs, RuleBook &rules, long *spatialSize) {
  Int batchSize = input_SGs.size();
  rules.clear();
  rules.resize(batchSize);
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; ++i) {
    lb[i] = 0;
    ub[i] = spatialSize[i] - 1;
  }
  auto region = RectangularRegion<dimension>(lb, ub);
  Int batchIdx;
#pragma omp parallel for private(batchIdx)
  for (batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    auto &iSG = input_SGs[batchIdx];
    for (auto const &inIter : iSG.mp) {
      rules[batchIdx].push_back(inIter.second + iSG.ctr);
      rules[batchIdx].push_back(region.offset(inIter.first));
    }
  }
}

#endif /* CONVOLUTIONRULES_H */
