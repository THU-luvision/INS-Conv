#include "hash_table.h"
#include "debugging.h"

#include <mt19937ar.h>

#include <cassert>

namespace CudaHT {
namespace CuckooHashing {

void GenerateFunctions(const unsigned  N,
                       const unsigned  num_keys,
                       const unsigned *d_keys,
                       const unsigned  table_size,
                             uint2    *constants) {
  bool regenerate = true;

  while (regenerate) {
    regenerate = false;

    // Generate a set of hash function constants for this build attempt.
    for (unsigned i = 0 ; i < N; ++i) {
      unsigned new_a = genrand_int32() % kPrimeDivisor;
      constants[i].x = (1 > new_a ? 1 : new_a);
      constants[i].y = genrand_int32() % kPrimeDivisor;
    }

#ifdef FORCEFULLY_GENERATE_NO_CYCLES
    // Ensure that every key gets N different slots.
    regenerate = CheckAssignedSameSlot(N, num_keys, d_keys, table_size, constants);
#endif
  }


#ifdef TAKE_HASH_FUNCTION_STATISTICS
  // Examine how well distributed the items are.
  TakeHashFunctionStatistics(num_keys, d_keys, table_size, constants, N);
#endif
}

}; // namespace CuckooHashing
}; // namespace CudaHT
