// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef Metadata_H
#define Metadata_H

#include "32bits.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <google/dense_hash_map>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <time.h>

#include <easy/profiler.h>
#include <easy/arbitrary_value.h>

#include <memory>
#include <assert.h>     /* assert */

// #include <torch/extension.h>

// #define PRINT_CONVOLUTION
// #define PRINT_SUBMANIFOLD
// #define PRINT_INPUTLAYER
// #define PRINT_CHUNK
// #define KERNEL_DEBUG 
 #define PRINT_NEW_SPTIAL_POINT

#define GPU_GRID
#define SUBMANIFOLD_CHUCK
#define NEW_SPTIAL_POINT

// #define MAX_INPUT_ADDRESS 3
#define MAX_INPUT_ADDRESS 320
// #define MAX_CHUNK_INPUT_COUNT 2

//#define PRINT_MEM_ALLOC


using namespace std;
template <Int dimension>
using Points = std::vector<Point<dimension>>;

using Ints = std::vector<Int>;

#include "../CUDA/CUDPPWrapper.hpp"
#include "../CUDA/RuleBookIterator.h"

template <Int dimension>
using SparseGridMap =
    google::dense_hash_map<Point<dimension>, Int, IntArrayHash<dimension>,
                           std::equal_to<Point<dimension>>>;
template <Int dimension> class SparseGrid {
public:
  Int ctr;
  SparseGridMap<dimension> mp;
  SparseGrid();
};
template <Int dimension> using SparseGrids = std::vector<SparseGrid<dimension>>;
using RuleBook = std::vector<std::vector<Int>>;

class Float3{
public:
float x;
float y;
float z;
  Float3                  ( void ){x = 0; y = 0; z = 0;}
  Float3                  ( const Float3& c ) {x = c.x; y = c.y; z = c.z;}
  Float3                  ( const float a, const float b, const float c ){x = a;y=b;z=c;}
  Float3                  ( const float* a ){x=a[0],y=a[1];z=a[2];}
  Float3                  ( const double* a ){x=a[0],y=a[1];z=a[2];}

  Float3 operator/   ( const Int n) const
  { return Float3(x/n,y/n,z/n); }
  Float3& operator/=   ( const Int n)
  { x/=n; y/=n ; z/=n; return *this; }
  Float3 operator+   ( const Float3& c ) const
  { return Float3(x+c.x,y+c.y,z+c.z); }
  Float3&         operator+=  ( const Float3& c )
  { x += c.x, y += c.y, z += c.z; return *this; }
  int             normalize            ( void )
  {
    float mag = sqrtf( x*x + y*y + z*z );
    if(mag < 1e-8) return 0;
    mag = 1 / mag; x *= mag; y *= mag; z *= mag;
    return 1;
  }

};

template <Int dimension> using Points = std::vector<Point<dimension>>;

/* GPU Sparse Grid */
// template <Int dimension>
// using Compacting_Point_Hashtable_ptr = std::shared_ptr<Compacting_Point_Hashtable<dimension>>;

template <Int dimension>
using Multival_Point_Hashtable_ptr = std::shared_ptr<Multival_Point_Hashtable<dimension>>;


template <Int dimension>
class GPU_SparseGrid    {
public:
  Int ctr;
  Multival_Point_Hashtable_ptr<dimension> pHash;
  GPU_SparseGrid();
};

template <Int dimension> using GPU_SparseGrids = std::vector<GPU_SparseGrid<dimension>>;

/* Submanifold Chuck */
class RBChunkTensor{
public:
    // gpu memory data
    at::Tensor inputAddress;
    at::Tensor outputAddress;
    at::Tensor rules;             // centered by output
    at::Tensor backward_rules; // centered by input

    void print()  {
      printf("inputAddress\n");
      // inputAddress.print();
      auto inputAddress_cpu = inputAddress.cpu();
      for(Int i = 0; i < inputAddress.sizes()[0]; i++) {
        printf("%d ", *(inputAddress_cpu.data<Int>()+i));
      }
      printf("\n");
      printf("outputAddress\n");
      auto outputAddress_cpu = outputAddress.cpu();
      for(Int i = 0; i < outputAddress.sizes()[0]; i++) {
        printf("%d ", *(outputAddress_cpu.data<Int>()+i));
      }
      printf("\n");
      printf("rules\n");
      auto rules_cpu = rules.cpu();
      for(Int i = 0; i < rules.sizes()[0]; i++) {
        printf("%d ", *(rules_cpu.data<Int>()+i));
      }
      printf("\n");
      printf("backward_rules\n");
      auto backward_rules_cpu = backward_rules.cpu();
      for(Int i = 0; i < backward_rules.sizes()[0]; i++) {
        printf("%d ", *(backward_rules_cpu.data<Int>()+i));
      }
      printf("\n");
    }
};

using RBChunkTensorList = std::vector<RBChunkTensor>;
class RBChunkPointerList {
public:
    std::vector<InputAddress> list;
    std::vector<Int*> cuda_mem_to_release;
    std::vector<short*> cuda_mem_to_release_short;

#ifdef PRINT_MEM_ALLOC
    long size = 0;
#endif

    RBChunkPointerList(){};

    ~RBChunkPointerList()
    {
#ifdef PRINT_MEM_ALLOC
  std::cout << "!!! RBChunkPointerList Deconstructor" << std::endl;
  // abort();
#endif
//        printf("~RBChunkPointerList()********************************************\nfree meme here\n*******************");
//        printf("\r\nFree RBChunk list\r\n");
        list.clear();
        while (!cuda_mem_to_release.empty()) {
            Int* tmp = cuda_mem_to_release.back();
            cuda_mem_to_release.pop_back();
            gpuErrchk(cudaFree(tmp));
        }
        while (!cuda_mem_to_release_short.empty()) {
            short* tmp = cuda_mem_to_release_short.back();
            cuda_mem_to_release_short.pop_back();
            gpuErrchk(cudaFree(tmp));
        }
    }
    void clear()
    {
        list.clear();
        while (!cuda_mem_to_release.empty()) {
            Int* tmp = cuda_mem_to_release.back();
            cuda_mem_to_release.pop_back();
            gpuErrchk(cudaFree(tmp));
        }
        while (!cuda_mem_to_release_short.empty()) {
            short* tmp = cuda_mem_to_release_short.back();
            cuda_mem_to_release_short.pop_back();
            gpuErrchk(cudaFree(tmp));
        }
    }
};

template <Int dimension>
void addPointToSparseGridMapAndFeatures(SparseGridMap<dimension> &mp,
                                        Point<dimension> p, Int &nActive,
                                        long nPlanes,
                                        /*float*/ at::Tensor features,
                                        float *vec, bool overwrite);

template <Int dimension> class Metadata {
public:
  int normal_guide_scale;
    // normal guided rotation invariant convolution filter
  std::unordered_map<Point<dimension>, std::vector<Float3>, IntArrayHash<dimension>>
      normals;
  // Count of active sites for each scale
  std::unordered_map<Point<dimension>, Int, IntArrayHash<dimension>> nActive;

  // Hash tables for each scale locating the active points
  std::unordered_map<Point<dimension>, SparseGrids<dimension>,
                     IntArrayHash<dimension>>
      grids;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      activePoolingRuleBooks;

  RuleBook inputLayerRuleBook;
  RuleBook blLayerRuleBook;

  std::unordered_map<Point<2 * dimension>, RuleBook,
                     IntArrayHash<2 * dimension>>
      submanifoldRuleBooks;
      
  std::unordered_map<Point<2 * dimension>, RuleBook,
                     IntArrayHash<2 * dimension>>
      incre_submanifoldRuleBooks;

  std::unordered_map<Point<dimension>, Int *,
                     IntArrayHash<dimension>>
      pre_exist_maps;

#ifdef SUBMANIFOLD_CHUCK
  // std::unordered_map<Point<2 * dimension>, RBChunkTensorList,
  //                    IntArrayHash<2 * dimension>>
  //     submanifoldRBChunkLists;
  std::unordered_map<Point<2 * dimension>, RBChunkPointerList,
                     IntArrayHash<2 * dimension>>
      submanifoldRBChunkPointerLists;

  std::unordered_map<Point<2 * dimension>, RBChunkPointerList,
                     IntArrayHash<2 * dimension>>
      incre_submanifoldRBChunkPointerLists;
#endif

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      permutohedralRuleBooks;

  std::unordered_map<Point<3 * dimension>, RuleBook,
                     IntArrayHash<3 * dimension>>
      ruleBooks;

  RuleBook fullConvolutionRuleBook;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      sparseToDenseRuleBooks;

  Point<dimension> inputSpatialSize;
  SparseGrids<dimension> *inputSGs;
  SparseGrid<dimension> *inputSG;
  Int *inputNActive;
  std::default_random_engine re;


#ifdef GPU_GRID
  GPU_SparseGrids<dimension> *GPU_inputSGs;
  GPU_SparseGrid<dimension> *GPU_inputSG;
  std::unordered_map<Point<dimension>, GPU_SparseGrids<dimension>,
                    IntArrayHash<dimension>>
    GPU_grids;

#endif

  Metadata();
  ~Metadata();
  void clear();
  Int getNActive(/*long*/ at::Tensor spatialSize);
  SparseGrids<dimension> &getSparseGrid(/*long*/ at::Tensor spatialSize);
  void setInputSpatialSize(/*long*/ at::Tensor spatialSize);
  void setNormalGuideScale(int scale_size)
  {
    normal_guide_scale = scale_size;
  }
  void batchAddSample();
  void setInputSpatialLocation(/*float*/ at::Tensor features,
                               /*long*/ at::Tensor location,
                               /*float*/ at::Tensor vec, bool overwrite);
  void setInputSpatialLocations(/*float*/ at::Tensor features,
                                /*long*/ at::Tensor locations,
                                /*float*/ at::Tensor vecs, bool overwrite);

  at::Tensor getSpatialLocations(/*long*/ at::Tensor spatialSize);
  void createMetadataForDenseToSparse(/*long*/ at::Tensor spatialSize,
                                      /*long*/ at::Tensor nz_, long batchSize);

  void sparsifyMetadata(Metadata<dimension> &mOut,
                        /*long*/ at::Tensor spatialSize,
                        /*byte*/ at::Tensor filter,
                        /*long*/ at::Tensor cuSum);

  void appendMetadata(Metadata<dimension> &mAdd,
                      /*long*/ at::Tensor spatialSize);

  std::vector<at::Tensor> sparsifyCompare(Metadata<dimension> &mReference,
                                          Metadata<dimension> &mSparsified,
                                          /*long*/ at::Tensor spatialSize);

  // tensor is size[0] x .. x size[dimension-1] x size[dimension]
  // size[0] x .. x size[dimension-1] == spatial volume
  // size[dimension] == #feature planes
  void addSampleFromThresholdedTensor(/*float*/ at::Tensor features_,
                                      /*float*/ at::Tensor tensor_,
                                      /*long*/ at::Tensor offset_,
                                      /*long*/ at::Tensor spatialSize_,
                                      float threshold);

  // 3x3 submanifold convolutions, 3x3/2x2 pooling or strided convolutions
  void generateRuleBooks3s2();

  // 3x3 submanifold convolutions, 2x2 pooling or strided convolutions
  void generateRuleBooks2s2();

  void inputLayer(/*long*/ at::Tensor spatialSize,
                  /*long*/ at::Tensor coords, Int batchSize, Int mode);
  void blLayer(/*long*/ at::Tensor spatialSize, /*long*/ at::Tensor coords,
               Int mode);
  RuleBook &getSubmanifoldRuleBook(/*long*/ at::Tensor spatialSize,
                                   /*long*/ at::Tensor size, bool openMP, int dilated_rate = 1);

  RuleBook &getIncreSubmanifoldRuleBook(
      /*long*/ at::Tensor spatialSize,
      /*long*/ at::Tensor size, bool openMP, int dilated_rate, Metadata<dimension>& pre_m);
  
  Int *getPreExistMaps(at::Tensor spatialSize, Metadata<dimension>& pre_m);

#ifdef SUBMANIFOLD_CHUCK
  RBChunkPointerList & getSubmanifoldChunkRuleBook(/*long*/ at::Tensor spatialSize,
                                                  /*long*/ at::Tensor size, bool openMP, int dilated_rate = 1);
  RBChunkPointerList & getIncreSubmanifoldChunkRuleBook(/*long*/ at::Tensor spatialSize,
                                                  /*long*/ at::Tensor size, bool openMP, int dilated_rate, Metadata<dimension>& pre_m);
#endif
  RuleBook &getPermutohedralSubmanifoldRuleBook(/*long*/ at::Tensor spatialSize,
                                                bool openMP);
  RuleBook &getActivePoolingRuleBook(/*long*/ at::Tensor spatialSize);
  RuleBook &getSparseToDenseRuleBook(/*long*/ at::Tensor spatialSize,
                                     bool openMP);
  RuleBook &getRuleBook(/*long*/ at::Tensor inputSpatialSize,
                        /*long*/ at::Tensor outputSpatialSize,
                        /*long*/ at::Tensor size,
                        /*long*/ at::Tensor stride, bool openMP);
  RuleBook &getFullConvolutionRuleBook(/*long*/ at::Tensor inputSpatialSize,
                                       /*long*/ at::Tensor outputSpatialSize,
                                       /*long*/ at::Tensor size,
                                       /*long*/ at::Tensor stride,
                                       Metadata<dimension> &newM);

  RuleBook &getRandomizedStrideRuleBook(/*long*/ at::Tensor inputSpatialSize,
                                        /*long*/ at::Tensor outputSpatialSize,
                                        /*long*/ at::Tensor size,
                                        /*long*/ at::Tensor stride,
                                        bool openMP);

  std::vector<at::Tensor>
  compareSparseHelper(Metadata<dimension> &mR,
                      /* long */ at::Tensor spatialSize);

  // at::Tensor
  // getSpatialLocations(/*long*/ at::Tensor spatialSize);
};

template <typename T> T *OptionalTensorData(at::Tensor tensor);

template <Int dimension> Int volume(long *point);
#endif
