/*! @file hash_functions.h
 *  @brief Hash function code.
 */

#ifndef HASH_FUNCTIONS__H
#define HASH_FUNCTIONS__H

#include <vector_types.h>
#include "definitions.h"

namespace CudaHT {
namespace CuckooHashing {

//! Prime number larger than the largest practical hash table size.
const unsigned kPrimeDivisor = 4294967291u;
// const Entry kPrimeDivisor = 4300000013lu;
// const unsigned kPrimeDivisor = 334214459;

//! Generates a set of linear hash function constants.
/*! @param[in]  N           Number of hash functions.
    @param[out] constants   CPU pointer to the constants.
    @param[in]  num_keys    Debug only: How many keys are in the input.
    @param[in]  d_keys      Debug only: Device memory array containing the input keys.
    @param[in]  table_size  Debug only: Size of the hash table.
 */
void GenerateFunctions(const unsigned  N,
                       const unsigned  num_keys,
                       const unsigned *d_keys,
                       const unsigned  table_size,
                             uint2    *constants);

//! Container for all of the hash functions.
template <unsigned N>
struct Functions {
  //! The constants required for all of the hash functions, including the stash.  Each function requires 2.
  uint2 constants[N];

  //! Generate new hash function constants.
  /*! The parameters are only used for debugging and examining the key distribution.
      \param[in] num_keys   Debug: Number of keys in the input.
      \param[in] d_keys     Debug: Device array of the input keys.
      \param[in] table_size Debug: Size of the hash table.
  */
  void Generate(const unsigned  num_keys,
                const unsigned *d_keys,
                const unsigned  table_size) {
    GenerateFunctions(N, num_keys, d_keys, table_size, constants);
  }
};

//! Computes the value of a hash function for a given key.
/*! \param[in] constants  Constants used by the hash function.
  ! \param[in] key        Key being hashed.
  ! \returns              The value of the hash function for the key.
 */
inline __device__ __host__
unsigned hash_function_inner(const uint2    constants,
                             const unsigned key) {
#if 0                             
  // Fast version.                             
  return ((constants.x ^ key) + constants.y) % kPrimeDivisor;
#else
  // Slow version.
  // return ((unsigned long long)constants.x * key + constants.y) % kPrimeDivisor;
  unsigned hash = constants.x;
  hash *= constants.y;
  hash ^= key & 0x000003FF;
  hash *= constants.y;
  hash ^= (key & 0x001FFC00) >> 10;
  hash *= constants.y;
  hash ^= (key & 0xFFE00000) >> 21;
  return hash;
#endif
}                             

//! Computes the value of a hash function for a given key.
/*! \param[in] functions        All of the constants used by the hash functions.
  ! \param[in] which_function   Which hash function is being used.
  ! \param[in] key              Key being hashed.
  ! \returns                    The value of a hash function with a given key.
 */
template <unsigned kNumHashFunctions>
inline __device__ __host__
unsigned hash_function(const Functions<kNumHashFunctions> functions,
                       const unsigned which_function,
                       const unsigned key) {
  return hash_function_inner(functions.constants[which_function], key);
}

//! Simple hash function used by the stash.
inline __device__ __host__
unsigned stash_hash_function(const uint2 stash_constants,
                             const unsigned key) {
  return (stash_constants.x ^ key + stash_constants.y) % kStashSize;
}


};  // namespace CuckooHashing
};  // namespace CudaHT

#endif
