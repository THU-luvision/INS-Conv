// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CUDA_RULEBOOKITERATOR_H
#define CUDA_RULEBOOKITERATOR_H


struct InputAddress
{
    void *aI; // address input
    void *aO; // address output
    void *outputRuleBook; // forward rb
    void *inputRuleBook;  // backward rb
    int cI; // input location num
    int cO; // output location num
      InputAddress() {}
    InputAddress(void* aI,
        void* aO,
        void* outputRuleBook,
        void* inputRuleBook,
        int cI,
        int cO)
    {
        this->aI = aI;
        this->aO = aO;
        this->outputRuleBook = outputRuleBook;
        this->inputRuleBook = inputRuleBook;
        this->cI = cI;
        this->cO = cO;
    }
};



// Macro to parallelize loading rulebook elements to CUDA memory and operating
// on the elements of the rulebook.
// X is the function to apply.
// Y is a command to run

#define RULEBOOKITERATOR(X, Y)                                                 \
  {                                                                            \
    Int rbMaxSize = 0;                                                         \
    for (auto &r : _rules)                                                     \
      rbMaxSize = std::max(rbMaxSize, (Int)r.size());                          \
    at::Tensor rulesBuffer = at::empty({rbMaxSize}, at::CUDA(at_kINT));        \
    Int *rbB = rulesBuffer.data<Int>();                                        \
    for (int k = 0; k < _rules.size(); ++k) {                                  \
      auto &r = _rules[k];                                                     \
      Int nHotB = r.size() / 2;                                                \
      if (nHotB) {                                                             \
        cudaMemcpy(rbB, &r[0], sizeof(Int) * 2 * nHotB,                        \
                   cudaMemcpyHostToDevice);                                    \
        X                                                                      \
      }                                                                        \
      Y                                                                        \
    }                                                                          \
  }

#endif /* CUDA_RULEBOOKITERATOR_H */
