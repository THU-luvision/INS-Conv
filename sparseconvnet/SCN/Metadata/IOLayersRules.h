// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "Metadata.h"
// Rulebook Format
// rules[0][0] == mode
// rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
// rules[0][2] == nInputRows
// rules[0][3] == nOutputRows
// rules[1]   nOutputRows x (1+maxActive)

// mode 0==guaranteed unique 1==overwrite, 2=keep, 3=sum, 4=mean
template <Int dimension>
void inputLayerRules(SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
                     Int nInputRows, Int nInputColumns, Int batchSize, Int mode,
                     Int &nActive) {

  EASY_FUNCTION(profiler::colors::Blue300);  

  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  SGs.resize(batchSize); // Set a minimum batch size if necessary
  Point<dimension> p;

  if (mode == 0) {
    nActive = nInputRows;
    rules.resize(1);
    rules[0].push_back(mode);
    rules[0].push_back(1);
    rules[0].push_back(nInputRows);
    rules[0].push_back(nInputRows);

    if (nInputColumns == dimension) {
      SGs.resize(1);
      auto &sg = SGs[0];
      for (Int i = 0; i < nInputRows; ++i) {
        for (Int j = 0; j < dimension; j++)
          p[j] = coords[j];
        coords += dimension;
        sg.mp[p] = i;
      }
    } else { // nInputColumns == dimension + 1
      Int idx;
      for (Int i = 0; i < nInputRows; ++i) {
        for (Int j = 0; j < dimension; j++)
          p[j] = coords[j];
        idx = coords[dimension];
        coords += dimension + 1;
        if (idx + 1 >= (Int)SGs.size())
          SGs.resize(idx + 1);
        SGs[idx].mp[p] = i;
      }
    }
    return;
  }

  // Compile list of how input rows correspond to output rows
  std::vector<std::vector<Int>> outputRows;
  if (nInputColumns == dimension) {
    SGs.resize(1);
    auto &sg = SGs[0];
    for (Int i = 0; i < nInputRows; ++i) {
      for (Int j = 0; j < dimension; j++)
        p[j] = coords[j];
      coords += dimension;
      auto iter = sg.mp.find(p);
      if (iter == sg.mp.end()) {
        sg.mp[p] = nActive++;
        outputRows.resize(nActive);
      }
      outputRows[sg.mp[p]].push_back(i);
    }
  } else { // nInputColumns == dimension + 1
    Int idx;
    for (Int i = 0; i < nInputRows; ++i) {
      for (Int j = 0; j < dimension; j++)
        p[j] = coords[j];
      idx = coords[dimension];
      coords += dimension + 1;
      if (idx + 1 >= (Int)SGs.size())
        SGs.resize(idx + 1);
      auto &sg = SGs[idx];
      auto iter = sg.mp.find(p);
      if (iter == sg.mp.end()) {
        sg.mp[p] = nActive++;
        outputRows.resize(nActive);
      }
      outputRows[sg.mp[p]].push_back(i);
    }
  }
  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(1); // replace with maxActive if mode==3 or 4
  rules[0].push_back(nInputRows);
  rules[0].push_back(outputRows.size());
  auto &rule = rules[1];
  if (mode == 1) {
    for (Int i = 0; i < nActive; ++i) {
      rule.push_back(1);
      rule.push_back(outputRows[i].front());
    }
  }
  if (mode == 2) {
    for (Int i = 0; i < nActive; ++i) {
      rule.push_back(1);
      rule.push_back(outputRows[i].back());
    }
  }
  if (mode == 3 or mode == 4) {
    Int maxActive = 0;
    for (auto &row : outputRows)
      maxActive = std::max(maxActive, (Int)row.size());
    rules[0][1] = maxActive;
    for (auto &row : outputRows) {
      rule.push_back(row.size());
      for (auto &r : row)
        rule.push_back(r);
      rule.resize((rule.size() + maxActive) / (maxActive + 1) *
                  (maxActive + 1));
    }
  }
}


#ifdef GPU_GRID

// TODO
template <Int dimension>
void inputLayerRulesSimple(GPU_SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
                     Int nInputRows, Int nInputColumns, Int batchSize, Int mode,
                     Int &nActive) {
  EASY_FUNCTION(profiler::colors::Blue300);

  assert(mode == 3 || mode == 4);
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  // Compile list of how input rows correspond to output rows
  // nInputColumns == dimension + 1
  Points<dimension> points;
  points.resize(nInputRows);
  SGs.resize(batchSize);
  std::vector<uint32_t> point_cloud_start(batchSize+1);

  uint32_t *d_keys = NULL;
  uint32_t *d_index = NULL;
  uint32_t *d_points = NULL;
  // Allocate memory for keys
  gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(uint32_t) * nInputRows));
  gpuErrchk(cudaMalloc((void**)&d_index, sizeof(uint32_t) * nInputRows));
  gpuErrchk(cudaMalloc((void**)&d_points, sizeof(uint32_t) * dimension * nInputRows));
  // get point_cloud_start(each batch start location), d_keys(input points hash key), d_index(d_index[i]=i),  d_points(nIn*3, pre nIn is x, mid is y, last nIn z)
  HashInputPointClouds<dimension>(coords, (uint32_t * )point_cloud_start.data(), d_points, d_keys, d_index, nInputRows, batchSize);
  Int numOutputRules = 0;
  int max_repeat_num = 0;
  for(Int i = 0; i < batchSize; i++)  {
    auto &SG = SGs[i];

    if(i != 0)  {
      SG.ctr = SGs[i-1].ctr + SGs[i-1].pHash->size;
    }

    int maxRepeat;
    // build SG input hash table
    SGs[i].pHash->InsertAndCompactPointCloud(d_keys + point_cloud_start[i],d_index + point_cloud_start[i],
                                                d_points,
                                                point_cloud_start[i+1] - point_cloud_start[i], nInputRows,maxRepeat);
    if(maxRepeat > max_repeat_num) max_repeat_num = maxRepeat;
    numOutputRules += SGs[i].pHash->size;
  }
  gpuErrchk(cudaFree(d_keys));
  gpuErrchk(cudaFree(d_index));
  gpuErrchk(cudaFree(d_points));
  std::vector<Int> currentRule(numOutputRules * (max_repeat_num + 1));
  Int * rule_index = (Int * )currentRule.data();


  for(Int i = 0; i < batchSize; i++)  {
    auto &SG = SGs[i];
    // build input rules, which map ouput location to input location.
    SG.pHash->generateInputRules(rule_index, max_repeat_num);
    rule_index += SG.pHash->size * (max_repeat_num+1);
  }

  nActive = numOutputRules;
  rules.clear();
  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(1); // replace with maxActive if mode==3 or 4
  rules[0].push_back(nInputRows);
  rules[0].push_back(numOutputRules);
  rules[1] = currentRule;
  rules[0][1] = max_repeat_num;


}



// TODO
template <Int dimension>
void inputLayerRules(GPU_SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
                     Int nInputRows, Int nInputColumns, Int batchSize, Int mode,
                     Int &nActive) {
  EASY_FUNCTION(profiler::colors::Blue300);
  EASY_BLOCK("coord to point");

  assert(mode == 3 || mode == 4);

  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  Point<dimension> p;

  // Compile list of how input rows correspond to output rows
  // nInputColumns == dimension + 1
  Points<dimension> points;
  points.resize(nInputRows);
  std::vector<Int> pc_start; // size = batchsize + 1
  int cnt = -1;
  long *coords_ptr = coords;

  for (Int i = 0; i < nInputRows; ++i) {
      for (Int j = 0; j < dimension; j++)
        p[j] = coords_ptr[j];
      points[i] = p;
      if(cnt < coords_ptr[dimension])
      {
          cnt ++;
          pc_start.push_back(i);
      }
      coords_ptr += (dimension+1);
  }
  pc_start.push_back(nInputRows);

  batchSize = pc_start.size() - 1;
  SGs.resize(batchSize); // Set a minimum batch size if necessary

  std::vector<std::vector<Int>> outputRows;
  EASY_END_BLOCK;
  EASY_BLOCK("Point to gpu hash");
  for(Int i = 0; i < batchSize; i++)  {
    auto &SG = SGs[i];
    Points<dimension> points_sample(points.begin() + pc_start[i], points.begin() + pc_start[i+1]);
    Ints idx;
    if(i != 0)  {
      SG.ctr = SGs[i-1].ctr + SGs[i-1].pHash->size;
    }
    // for(Int k = 0; k < (Int)points_sample.size(); k++)  {
    //   // std::cout << points_sample[k] << std::endl;
    //     printf("%d, %d, %d\n", points_sample[k][0], points_sample[k][1], points_sample[k][2]);
    // }
    SG.pHash->insert_points(points_sample);
    SG.pHash->retrieve_points(points_sample, idx);

    outputRows.resize(SG.ctr + SGs[i].pHash->size);
    for(Int j = 0; j < (Int)idx.size(); j++) {
      outputRows[SG.ctr + idx[j]].push_back(pc_start[i] + j);
    }
  }
  EASY_END_BLOCK;
  EASY_BLOCK("max,mean");
  nActive = outputRows.size();

  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(1); // replace with maxActive if mode==3 or 4
  rules[0].push_back(nInputRows);
  rules[0].push_back(outputRows.size());
  auto &rule = rules[1];

  if (mode == 3 or mode == 4) {
    Int maxActive = 0;
    for (auto &row : outputRows)
      maxActive = std::max(maxActive, (Int)row.size());
    rules[0][1] = maxActive;
    for (auto &row : outputRows) {
      rule.push_back(row.size());
      for (auto &r : row)
        rule.push_back(r);
      rule.resize((rule.size() + maxActive) / (maxActive + 1) *
                  (maxActive + 1));
    }
  }
  EASY_END_BLOCK;
}

#endif

// Rulebook Format
// rules[0][0] == mode
// rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
// rules[0][2] == batchSize
// rules[0][3] == length
// rules[0][4] == nOutputRows
// rules[1]   nOutputRows x (1+maxActive)

// bl is a batchSize x length x dimension long array of coordinates
// mode 0==guaranteed unique and all present; 1==overwrite, 2=keep, 3=sum,
// 4=mean
template <Int dimension>
void blRules(SparseGrids<dimension> &SGs, RuleBook &rules, long *coords,
             Int batchSize, Int length, Int mode, Int &nActive) {
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);
  SGs.resize(batchSize);
  Int I;

  if (mode == 0) {
    nActive = batchSize * length;
    rules.resize(1);
    rules[0].push_back(mode);
    rules[0].push_back(1);
    rules[0].push_back(batchSize);
    rules[0].push_back(length);
    rules[0].push_back(nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &sg = SGs[I];
      sg.ctr = I * length;
      auto c = coords + I * length * dimension;
      Point<dimension> p;
      for (Int l = 0; l < length; ++l) {
        for (Int j = 0; j < dimension; ++j)
          p[j] = c[j];
        c += dimension;
        sg.mp[p] = l;
      }
    }
    return;
  }

  // Compile list of how input rows correspond to output rows
  std::vector<std::vector<std::vector<Int>>> outputRows(batchSize);
  std::vector<Int> nActives(batchSize);
#pragma omp parallel for private(I)
  for (I = 0; I < batchSize; I++) {
    auto &sg = SGs[I];
    auto &ors = outputRows[I];
    auto &nAct = nActives[I];
    auto c = coords + I * length * dimension;
    Int i = I * length;
    Point<dimension> p;
    for (Int l = 0; l < length; ++l, ++i) {
      for (Int j = 0; j < dimension; ++j)
        p[j] = *c++;
      if (p[0] >= 0) {
        auto iter = sg.mp.find(p);
        if (iter == sg.mp.end()) {
          sg.mp[p] = nAct++;
          ors.resize(nAct);
        }
        ors[sg.mp[p]].push_back(i);
      }
    }
  }

  for (I = 0; I < batchSize; I++) {
    SGs[I].ctr = nActive;
    nActive += nActives[I];
  }
  Int maxActive = 1;
  if (mode >= 3)
    for (auto &ors : outputRows)
      for (auto &row : ors)
        maxActive = std::max(maxActive, (Int)row.size());

  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(maxActive);
  rules[0].push_back(batchSize);
  rules[0].push_back(length);
  rules[0].push_back(nActive);
  auto &rule = rules[1];
  if (mode == 1) {
    rule.resize(2 * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * 2];
      for (auto &row : ors) {
        rr[0] = row.size();
        rr[1] = row.back();
        rr += 2;
      }
    }
  }
  if (mode == 2) {
    rule.resize(2 * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * 2];
      for (auto &row : ors) {
        rr[0] = row.size();
        rr[1] = row.front();
        rr += 2;
      }
    }
  }
  if (mode == 3 or mode == 4) {
    rule.resize((maxActive + 1) * nActive);
#pragma omp parallel for private(I)
    for (I = 0; I < batchSize; I++) {
      auto &ors = outputRows[I];
      auto rr = &rule[SGs[I].ctr * (maxActive + 1)];
      for (auto &row : ors) {
        rr[0] = row.size();
        for (Int i = 0; i < (Int)row.size(); ++i)
          rr[i + 1] = row[i];
        rr += 1 + maxActive;
      }
    }
  }
}

#endif /* INPUTLAYER_H */
