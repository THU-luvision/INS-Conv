// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef SUBMANIFOLDCONVOLUTIONRULES_H
#define SUBMANIFOLDCONVOLUTIONRULES_H
#include "Metadata.h"
#include <cmath>
#include <google/dense_hash_map>
#include "../CUDA/SubmanifoldRules_cuda.cpp"

// Full input region for an output point
template <Int dimension>
RectangularRegion<dimension>
InputRegionCalculator_Submanifold(const Point<dimension> &output, long *size) {
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    Int pad = size[i] / 2;
    lb[i] = output[i] - pad;
    ub[i] = output[i] + size[i] - 1 - pad;
  }
  return RectangularRegion<dimension>(lb, ub);
}


template <Int dimension> using PointVector = std::vector<Point<dimension> >;
template <Int dimension>
void printPoint(Point<dimension> &p)
{
    for(int i = 0; i < dimension; i++)
    {
        printf("%d ", p[i]);
    }
    printf("\r\n");
}
template <Int dimension>
void getCandidates(Int D, PointVector<dimension> &elements, Point<dimension> &p, int dilated_rate)
{
    if(D < 1) return;
    for(int i = -1; i <= 1; i++)
    {
        Point<dimension> np = p;
        np[D-1] += i * dilated_rate;
        getCandidates<dimension>(D-1,elements,np,dilated_rate);
        if(D == 1)
        {
            elements.push_back(np);
        }
    }
}

// Dilated convolution for input region
// std vector type return, but also could be used for iteration
// Only supports 3 dimension currenctly.
template <Int dimension>
PointVector<dimension> Candidates_SubmanifoldDialted(const Point<dimension> &output, long *size, int dilated_rate) {
  PointVector<dimension> candidates;
  Int elementNum = Int(std::pow(3,dimension));
  candidates.reserve(elementNum);
  Point<dimension> p = output;
  getCandidates<dimension>(dimension,candidates,p,dilated_rate);
//  printf("\r\n\r\n");
//  for(int i = 0; i < candidates.size();i++)
//  {
//      printPoint<dimension>(candidates[i]);
//  }
  return candidates;
#if 0
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    Int pad = size[i] / 2;
    lb[i] = output[i] - pad;
    ub[i] = output[i] + size[i] - 1 - pad;
  }
  return RectangularRegion<dimension>(lb, ub);
  #endif
}
// 1 for success
// 0 for fail
template <Int dimension>
int NearestNeighborSearch(const Point<dimension> &inputPoint,
            int range, SparseGrid<dimension> &grid, Int &loc)
{
      for(int i = -range; i <= range; i++)
      {
             for(int j = -range; j <= range; j++)
             {
                  for(int k = -range; k <= range; k++)
                  {
                    if(i == range || i == -range || j == range || j == -range || k == range || k == -range)
                    {
                        auto inputIter = grid.mp.find(inputPoint);
                        if (inputIter != grid.mp.end())
                        {
                            loc = inputIter->second;
                            return 1;
                        }
                    }

                  }
             }
      }
      return 0;
}




// Call for each convolutional / max-pooling layer, once for each batch item.
// rules is used to carry out the "lowering" whilst carrying out the convolution

template <Int dimension>
double SubmanifoldConvolution_SgToRules(SparseGrid<dimension> &grid,
                                        RuleBook &rules, long *size, int dilated_rate = 1) {
  double countActiveInputs = 0;
  for (auto const &outputIter : grid.mp) {
//    auto inRegion = InputRegionCalculator_Submanifold<dimension>(outputIter.first, size);
    auto inRegion = Candidates_SubmanifoldDialted<dimension>(outputIter.first, size, dilated_rate);
    Int rulesOffset = 0;
    for (auto inputPoint : inRegion) {

    // input point local search for better information aggregation
      Int loc = -1;
      assert(dimension == 3);
      if(dilated_rate > 1)
      {
          int range = floor(dilated_rate / 2);
          for(int i = 0; i <= range; i++)
          {
            int search = NearestNeighborSearch<dimension>(inputPoint,i,grid,loc);
            if(search == 1) break;
          }
      }
      else
      {
          auto inputIter = grid.mp.find(inputPoint);
          if(inputIter != grid.mp.end()) loc = inputIter->second;
      }
      if (loc >= 0) {
        rules[rulesOffset].push_back(loc + grid.ctr);
        rules[rulesOffset].push_back(outputIter.second + grid.ctr);
        countActiveInputs++;
      }
      rulesOffset++;
    }
  }
  return countActiveInputs;
}

// Call for each convolutional / max-pooling layer, once for each batch item.
// rules is used to carry out the "lowering" whilst carrying out the convolution

template <Int dimension>
double SubmanifoldConvolution_SgToRules(SparseGrid<dimension> &grid,
                                        RuleBook &rules, long *size, 
					const std::vector<Float3> &normal, int dilated_rate = 1) 
{
  EASY_FUNCTION(profiler::colors::Amber200);
  Int index[27 * 6] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
                        24,25,26,21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,5,0,1,2,
                        6,7,8,15,16,17,24,25,26,3,4,5,12,13,14,21,22,23,0,1,2,9,10,11,18,19,20,
                        18,19,20,9,10,11,0,1,2,21,22,23,12,13,14,3,4,5,24,25,26,15,16,17,6,7,8,
                        2,11,20,5,14,23,8,17,26,1,10,19,4,13,22,7,16,25,0,9,18,3,12,21,6,15,24,
                        18,9,0,21,12,3,24,15,6,19,10,1,22,13,4,25,16,7,20,11,2,23,14,5,26,17,8
                        };


  Int NActivePoints = grid.mp.size();
  EASY_VALUE("NActivePoints", NActivePoints);

  double countActiveInputs = 0;
  for (auto const &outputIter : grid.mp) {
    auto inRegion =
	InputRegionCalculator_Submanifold<dimension>(outputIter.first, size);
    Int pointID = outputIter.second;
    const Float3 &n = normal[pointID];

    Int rulesOffset = 0;
    Int oriIndex = OrientedFilter(n);
    Int *conv_rule_index = &index[oriIndex*27];
    for (auto inputPoint : inRegion) {
    // input point local search for better information aggregation
      Int loc = -1;
      assert(dimension == 3);
      if(dilated_rate > 1)
      {
          int range = floor(dilated_rate / 2);
          for(int i = 0; i <= range; i++)
          {
            int search = NearestNeighborSearch<dimension>(inputPoint,i,grid,loc);
            if(search == 1) break;
          }
      }
      else
      {
          auto inputIter = grid.mp.find(inputPoint);
          if(inputIter != grid.mp.end()) loc = inputIter->second;
      }
      if (loc >= 0) {
        Int ruleIndex = conv_rule_index[rulesOffset];
        rules[ruleIndex].push_back(loc + grid.ctr);
        rules[ruleIndex].push_back(outputIter.second + grid.ctr);
        countActiveInputs++;
      }
      rulesOffset++;
    }
  }
  return countActiveInputs;
}

template <Int dimension>
void remap_rules_with_normal(RuleBook & rules, const std::vector<Float3> &normal)
{
    EASY_FUNCTION(profiler::colors::Amber200);
    Int index[27 * 6] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
                          24,25,26,21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,5,0,1,2,
                          6,7,8,15,16,17,24,25,26,3,4,5,12,13,14,21,22,23,0,1,2,9,10,11,18,19,20,
                          18,19,20,9,10,11,0,1,2,21,22,23,12,13,14,3,4,5,24,25,26,15,16,17,6,7,8,
                          2,11,20,5,14,23,8,17,26,1,10,19,4,13,22,7,16,25,0,9,18,3,12,21,6,15,24,
                          18,9,0,21,12,3,24,15,6,19,10,1,22,13,4,25,16,7,20,11,2,23,14,5,26,17,8
                          };
    std::vector<Int> oriIndex(normal.size());
    for(size_t i =0; i < normal.size(); i++)
    {
        const Float3 &n = normal[i];
        oriIndex[i] = OrientedFilter(n);
    }

    RuleBook new_rules;
    new_rules.resize(27);
    for(int i = 0; i < 27; i++)
    {
        std::vector<Int> & old_rule = rules[i];
        for(size_t j = 0; j < old_rule.size();j+=2)
        {
            Int new_rule_index = index[oriIndex[old_rule[j+1]] * 27 + i];
            std::vector<Int> & new_rule = new_rules[new_rule_index];
            new_rule.push_back(old_rule[j]);
            new_rule.push_back(old_rule[j+1]);
        }
    }
    rules = new_rules;
}

#ifdef GPU_GRID
template <Int dimension>
double SubmanifoldConvolution_SgToRules(GPU_SparseGrid<dimension> &gpu_grid,
                                        RuleBook &rules, long *size, int dilated_rate = 1) 
{
  EASY_FUNCTION(profiler::colors::Amber200);

  Int NActivePoints = gpu_grid.pHash->size;
  EASY_VALUE("NActivePoints", NActivePoints);
#ifdef PRINT_SUBMANIFOLD
  printf("NActivePoints = %d\n", NActivePoints);
#endif

  Int countActiveInputs = 0;

  SubmanifoldBuildObject<dimension> build(NActivePoints, size, gpu_grid);

  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  build.generateQueryList();
  EASY_END_BLOCK;
  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  build.global_hash_query();
  EASY_END_BLOCK;
  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  countActiveInputs += build.generate_rulebook(rules);
  EASY_END_BLOCK;
  return (double)countActiveInputs;
}

template <Int dimension>
double Incre_SubmanifoldConvolution_SgToRules(GPU_SparseGrid<dimension> &gpu_grid,
                                        RuleBook &rules, long *size, int dilated_rate, GPU_SparseGrid<dimension> &pre_SG) 
{
  EASY_FUNCTION(profiler::colors::Amber200);

  Int NActivePoints = gpu_grid.pHash->size;
  EASY_VALUE("NActivePoints", NActivePoints);
#ifdef PRINT_SUBMANIFOLD
  printf("NActivePoints = %d\n", NActivePoints);
#endif

  Int countActiveInputs = 0;

  IncreSubmanifoldBuildObject<dimension> build(NActivePoints, size, gpu_grid, pre_SG);

  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  build.generateQueryList();
  EASY_END_BLOCK;
  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  build.global_hash_query();
  EASY_END_BLOCK;
  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  countActiveInputs += build.generate_rulebook(rules);
  EASY_END_BLOCK;
  return (double)countActiveInputs;
}


#endif

#ifdef SUBMANIFOLD_CHUCK
template <Int dimension>
double SubmanifoldConvolution_SgToRules_Chuck(GPU_SparseGrid<dimension> &gpu_grid,
                                              const std::vector<Float3> &normal,
                                              RBChunkPointerList& new_rules, long *size, int dilated_rate = 1) 
{
  EASY_FUNCTION(profiler::colors::Amber200);

  Int NActivePoints = gpu_grid.pHash->size;
  EASY_VALUE("NActivePoints", NActivePoints);
#ifdef PRINT_SUBMANIFOLD
  printf("NActivePoints = %d\n", NActivePoints);
#endif

  Int countActiveOutputs = 0;

  // Level 0 : 16
  EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100

  SubmanifoldChuckBuildObject<dimension> build(NActivePoints, size, 16, gpu_grid, 1);

  build.generateQueryList();
  build.global_hash_query();
  build.get_block_id();
  build.get_ori_index(normal);
  build.get_chunk_input_address();

  countActiveOutputs += build.generate_rulebook(new_rules);
  EASY_END_BLOCK;

  // SubmanifoldChuckBuildObject<dimension> build_level2(build, 4);
  // if(build_level2.NActivePoints > 0)  {
  //         EASY_BLOCK("Level 2: 4"); // Begin block with default color == Amber100
  //   build_level2.generateQueryList();
  //   build_level2.global_hash_query();
  //   build_level2.get_block_id();
  //   build_level2.get_chunk_input_address();
  //   countActiveOutputs += build_level2.generate_rulebook(new_rules);
  //   EASY_END_BLOCK;

  // }
#if 1
  // Level 1: 8

  SubmanifoldChuckBuildObject<dimension> build_level1(build, 8);
  if(build_level1.NActivePoints > 0)  {
  EASY_BLOCK("Level 1 : 8"); // Begin block with default color == Amber100

    build_level1.generateQueryList();
    build_level1.global_hash_query();
    build_level1.get_block_id();
    // build_level1.get_ori_index(normal);
    build_level1.get_chunk_input_address();
    countActiveOutputs += build_level1.generate_rulebook(new_rules);
    EASY_END_BLOCK;

    // Level 2: 4

    SubmanifoldChuckBuildObject<dimension> build_level2(build_level1, 4);
    if(build_level2.NActivePoints > 0)  {
      EASY_BLOCK("Level 2 : 4"); // Begin block with default color == Amber100

      build_level2.generateQueryList();
      build_level2.global_hash_query();
      build_level2.get_block_id();
      // build_level2.get_ori_index(normal);
      build_level2.get_chunk_input_address();
      countActiveOutputs += build_level2.generate_rulebook(new_rules);
      EASY_END_BLOCK;

    }
  }
#endif

  if(countActiveOutputs != NActivePoints) {
    printf("!!! Split Error, countActiveOutputs = %d\n", countActiveOutputs);
    abort();
  }

  return countActiveOutputs;
}

template <Int dimension>
double SubmanifoldConvolution_SgToRules_Chuck(GPU_SparseGrid<dimension> &gpu_grid,
                                              RBChunkPointerList& new_rules, long *size, int dilated_rate = 1) 
{
  EASY_FUNCTION(profiler::colors::Amber200);

  Int NActivePoints = gpu_grid.pHash->size;
  EASY_VALUE("NActivePoints", NActivePoints);
#ifdef PRINT_SUBMANIFOLD
  printf("NActivePoints = %d\n", NActivePoints);
#endif

  Int countActiveOutputs = 0;

  // Level 0 : 16
    EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  SubmanifoldChuckBuildObject<dimension> build(NActivePoints, size, 8, gpu_grid, 0);

  build.generateQueryList();
  build.global_hash_query();
  build.get_block_id();
  build.get_chunk_input_address();

  countActiveOutputs += build.generate_rulebook(new_rules);
  EASY_END_BLOCK;

   SubmanifoldChuckBuildObject<dimension> build_level2(build, 4);
   if(build_level2.NActivePoints > 0)  {
           EASY_BLOCK("Level 2: 4"); // Begin block with default color == Amber100
     build_level2.generateQueryList();
     build_level2.global_hash_query();
     build_level2.get_block_id();
     build_level2.get_chunk_input_address();
     countActiveOutputs += build_level2.generate_rulebook(new_rules);
     EASY_END_BLOCK;

   }
#if 0
  // Level 1: 8

  SubmanifoldChuckBuildObject<dimension> build_level1(build, 8);
  if(build_level1.NActivePoints > 0)  {
    EASY_BLOCK("Level 1: 8"); // Begin block with default color == Amber100

    build_level1.generateQueryList();
    build_level1.global_hash_query();
    build_level1.get_block_id();
    build_level1.get_chunk_input_address();
    countActiveOutputs += build_level1.generate_rulebook(new_rules);
    EASY_END_BLOCK;

    // Level 2: 4

    SubmanifoldChuckBuildObject<dimension> build_level2(build_level1, 4);
    if(build_level2.NActivePoints > 0)  {
            EASY_BLOCK("Level 2: 4"); // Begin block with default color == Amber100
      build_level2.generateQueryList();
      build_level2.global_hash_query();
      build_level2.get_block_id();
      build_level2.get_chunk_input_address();
      countActiveOutputs += build_level2.generate_rulebook(new_rules);
      EASY_END_BLOCK;

    }
  }
#endif
  if(countActiveOutputs != NActivePoints) {
    printf("!!! Split Error, countActiveOutputs = %d\n", countActiveOutputs);
    abort();
  }

  return countActiveOutputs;
}

template <Int dimension>
double Incre_SubmanifoldConvolution_SgToRules_Chuck(GPU_SparseGrid<dimension> &gpu_grid,
                                              RBChunkPointerList& new_rules, long *size, int dilated_rate, GPU_SparseGrid<dimension>&pre_grid) 
{
  EASY_FUNCTION(profiler::colors::Amber200);

  Int NActivePoints = gpu_grid.pHash->size;
  EASY_VALUE("NActivePoints", NActivePoints);
#ifdef PRINT_SUBMANIFOLD
  printf("NActivePoints = %d\n", NActivePoints);
#endif

  Int countActiveOutputs = 0;

  // Level 0 : 16
    EASY_BLOCK("Level 0 : 8"); // Begin block with default color == Amber100
  IncreSubmanifoldChuckBuildObject<dimension> build(NActivePoints, size, 8, gpu_grid, 0, pre_grid);
  build.generateQueryList();
  build.global_hash_query();
  build.get_block_id();
  build.get_chunk_input_address();
  countActiveOutputs += build.generate_rulebook(new_rules);
  if (build.NActivePoints == 0) {
      InputAddress rbp = InputAddress(
      NULL,
      NULL,
      NULL,
      NULL,
      0,
      0);
      new_rules.list.push_back(rbp);
      return 0;
  } 


  EASY_END_BLOCK;
  IncreSubmanifoldChuckBuildObject<dimension> build_level2(build, 4);
  if(build_level2.NActivePoints > 0)  {
    EASY_BLOCK("Level 2: 4"); // Begin block with default color == Amber100
    build_level2.generateQueryList();
    build_level2.global_hash_query();
    build_level2.get_block_id();
    build_level2.get_chunk_input_address();
    countActiveOutputs += build_level2.generate_rulebook(new_rules);
    EASY_END_BLOCK;
  }

   

  // if(countActiveOutputs != NActivePoints) {
  //   printf("!!! Split Error, countActiveOutputs = %d\n", countActiveOutputs);
  //   abort();
  // }

  return countActiveOutputs;
}

#endif


template <Int dimension>
Int SubmanifoldConvolution_SgsToRules(
#ifdef GPU_GRID
                                      GPU_SparseGrids<dimension> &SGs,
#else
                                      SparseGrids<dimension> &SGs,
#endif
                                      RuleBook &rules, long *size, int dilated_rate = 1) {
  
  EASY_FUNCTION(profiler::colors::Amber100);
  
  Int sd = volume<dimension>(size);
  Int countActiveInputs = 0;
  rules.clear();
  rules.resize(sd);
  for (Int i = 0; i < (Int)SGs.size(); i++)  {  // Loop over batch size
    countActiveInputs +=
        SubmanifoldConvolution_SgToRules<dimension>(SGs[i], rules, size, dilated_rate);
  }
  
  // Debug: Print rulebook
#ifdef PRINT_SUBMANIFOLD
  printf("Submanifold rules:\n");
  for (Int i = 0; i < (Int)rules.size(); i++) { 
    for (Int j = 0; j < (Int)rules[i].size(); j+=2)  {
      std::cout << "Offset: " << i << ", Rules: " << rules[i][j] << ", "<< rules[i][j+1] << std::endl; 
    }
    std::cout << std::endl; 
  } 
  printf("countActiveInputs = %d\n", countActiveInputs);
#endif

  return countActiveInputs;
}



template <Int dimension>
Int Incre_SubmanifoldConvolution_SgsToRules(
#ifdef GPU_GRID
                                      GPU_SparseGrids<dimension> &SGs,
#else
                                      SparseGrids<dimension> &SGs,
#endif
                                      RuleBook &rules, long *size, int dilated_rate, GPU_SparseGrids<dimension> &pre_SGs) {
  
  EASY_FUNCTION(profiler::colors::Amber100);
  
  Int sd = volume<dimension>(size);
  Int countActiveInputs = 0;
  rules.clear();
  rules.resize(sd);
  for (Int i = 0; i < (Int)SGs.size(); i++)  {  // Loop over batch size
    countActiveInputs +=
        Incre_SubmanifoldConvolution_SgToRules<dimension>(SGs[i], rules, size, dilated_rate, pre_SGs[i]);
  }
  
  // Debug: Print rulebook
#ifdef PRINT_SUBMANIFOLD
  printf("Submanifold rules:\n");
  for (Int i = 0; i < (Int)rules.size(); i++) { 
    for (Int j = 0; j < (Int)rules[i].size(); j+=2)  {
      std::cout << "Offset: " << i << ", Rules: " << rules[i][j] << ", "<< rules[i][j+1] << std::endl; 
    }
    std::cout << std::endl; 
  } 
  printf("countActiveInputs = %d\n", countActiveInputs);
#endif

  return countActiveInputs;
}

template <Int dimension>
Int SubmanifoldConvolution_SgsToRules(
#ifdef GPU_GRID
                                      GPU_SparseGrids<dimension> &SGs,
#else
                                      SparseGrids<dimension> &SGs,
#endif
                                      RuleBook &rules, 
                                      long *size, const std::vector<Float3> &normal,int dilated_rate = 1) {
  
  EASY_FUNCTION(profiler::colors::Amber100);
  Int sd = volume<dimension>(size);
  Int countActiveInputs = 0;
  rules.clear();
  rules.resize(sd);
  for (Int i = 0; i < (Int)SGs.size(); i++)
  {
    countActiveInputs +=
        SubmanifoldConvolution_SgToRules<dimension>(SGs[i], rules, size, dilated_rate);
  }
  remap_rules_with_normal<dimension>(rules,normal);
  // Debug: Print rulebook
#ifdef PRINT_SUBMANIFOLD
  printf("Submanifold rules with normal:\n");
  for (Int i = 0; i < (Int)rules.size(); i++) { 
    for (Int j = 0; j < (Int)rules[i].size(); j+=2)  {
      std::cout << "Offset: " << i << ", Rules: " << rules[i][j] << ", "<< rules[i][j+1] << std::endl; 
    }
    std::cout << std::endl; 
  } 
  printf("countActiveInputs = %d\n", countActiveInputs);
#endif
  
  return countActiveInputs;
}

#ifdef SUBMANIFOLD_CHUCK

/* Chuck with normal */
template <Int dimension>
Int SubmanifoldConvolution_SgsToRules(GPU_SparseGrids<dimension> &SGs,
                                      RBChunkPointerList& new_rules,
                                      long *size,int dilated_rate = 1) {
  
  EASY_FUNCTION(profiler::colors::Amber100);
  Int countActiveInputs = 0;
  new_rules.clear();
  for (Int i = 0; i < (Int)SGs.size(); i++)
  {
    SubmanifoldConvolution_SgToRules_Chuck<dimension>(SGs[i], new_rules, size, dilated_rate);
  }
  // Debug: Print rulebook
#if 0
// #ifdef PRINT_SUBMANIFOLD
  printf("Chuck rules with normal:\n");
  printf("Total blocks = %d\n", (Int)rules.size());
  for(size_t i = 0; i < rules.size(); i++) {
    printf("Block %d\n", (Int)i);
    rules[i].print();
  }
#endif

#ifdef PRINT_CHUNK
  for(size_t i = 0; i < rules.size(); i++) {
    auto &rb = rules[i];
    
    printf("Block : %d\n", (Int)i);

    rb.print();

  }
  abort();
#endif

  return countActiveInputs;
}

/* Chuck with normal */
template <Int dimension>
Int Incre_SubmanifoldConvolution_SgsToRules(GPU_SparseGrids<dimension> &SGs,
                                      RBChunkPointerList& new_rules,
                                      long *size,int dilated_rate , GPU_SparseGrids<dimension>& pre_SGs) {
  
  EASY_FUNCTION(profiler::colors::Amber100);
  Int countActiveInputs = 0;
  new_rules.clear();
  for (Int i = 0; i < (Int)SGs.size(); i++)
  {
    Incre_SubmanifoldConvolution_SgToRules_Chuck<dimension>(SGs[i], new_rules, size, dilated_rate, pre_SGs[i]);
  }
  // Debug: Print rulebook
#if 0
// #ifdef PRINT_SUBMANIFOLD
  printf("Chuck rules with normal:\n");
  printf("Total blocks = %d\n", (Int)rules.size());
  for(size_t i = 0; i < rules.size(); i++) {
    printf("Block %d\n", (Int)i);
    rules[i].print();
  }
#endif

#ifdef PRINT_CHUNK
  for(size_t i = 0; i < rules.size(); i++) {
    auto &rb = rules[i];
    
    printf("Block : %d\n", (Int)i);

    rb.print();

  }
  abort();
#endif

  return countActiveInputs;
}


/* chuck without normal*/
template <Int dimension>
Int SubmanifoldConvolution_SgsToRules(GPU_SparseGrids<dimension> &SGs,
                                      RBChunkPointerList& new_rules,
                                      long *size, const std::vector<Float3> &normal,int dilated_rate = 1) {
  EASY_FUNCTION(profiler::colors::Amber100);
  Int countActiveInputs = 0;
  new_rules.clear();
  for (Int i = 0; i < (Int)SGs.size(); i++)
  {
    SubmanifoldConvolution_SgToRules_Chuck<dimension>(SGs[i], normal, new_rules, size, dilated_rate);
  }
  // Debug: Print rulebook
#if 0
// #ifdef PRINT_SUBMANIFOLD
  printf("Chuck rules with normal:\n");
  printf("Total blocks = %d\n", (Int)rules.size());
  for(size_t i = 0; i < rules.size(); i++) {
    printf("Block %d\n", (Int)i);
    rules[i].print();
  }
#endif

#ifdef PRINT_CHUNK
  for(size_t i = 0; i < rules.size(); i++) {
    auto &rb = rules[i];
    
    printf("Block : %d\n", (Int)i);

    rb.print();

  }
  abort();
#endif

  return countActiveInputs;
}
#endif

template <Int dimension>
Int SubmanifoldConvolution_SgsToRules_OMP(SparseGrids<dimension> &SGs,
                                          RuleBook &rules, long *size) {
  // py::print("\tSgsToRules OMP\n");
  std::vector<RuleBook> rbs(SGs.size());
  std::vector<double> countActiveInputs(SGs.size());
  rules.clear();
  Int sd = volume<dimension>(size);
  rules.resize(sd);
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < (Int)SGs.size(); i++) {
      rbs[i].resize(sd);
      countActiveInputs[i] =
          SubmanifoldConvolution_SgToRules<dimension>(SGs[i], rbs[i], size);
    }
  }
  {
    Int i;
#pragma omp parallel for private(i)
    for (i = 0; i < sd; i++)
      for (auto const &rb : rbs)
        rules[i].insert(rules[i].end(), rb[i].begin(), rb[i].end());
  }
  Int countActiveInputs_ = 0;
  for (auto &i : countActiveInputs)
    countActiveInputs_ += i;
  return countActiveInputs_;
}

#endif /* SUBMANIFOLDCONVOLUTIONRULES_H */
