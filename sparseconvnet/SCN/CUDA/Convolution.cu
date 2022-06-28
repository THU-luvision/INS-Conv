// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RuleBookIterator.h"
#include "../Metadata/Metadata.h"
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "assert.h"
#define TACC double

#define DEBUG_FAST_BACKWARD 0
#define DEBUG_FAST_CONV 0

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
#include <easy/profiler.h>

#define TACC double

template <typename T>
__global__ void Convolution_fp_bias_(T *output_features, T *bias, Int nPlanes,
				     Int nActive) {
  Int n = blockIdx.x * 32 + threadIdx.x;
  T b = bias[n];
  output_features += n;
  for (Int row = blockIdx.y; row < nActive; row += gridDim.y) {
    output_features[row * nPlanes] = b;
  }
}

template <typename T>
void Convolution_fp_bias(T *oF, T *b, Int nPlanes, Int nActive) {
  if (nPlanes / 32 > 0)
    Convolution_fp_bias_<<<dim3(nPlanes / 32, 4096), 32>>>(oF, b, nPlanes,
							   nActive);
  if (nPlanes % 32 > 0) {
    Int o = nPlanes / 32 * 32;
    Convolution_fp_bias_<<<dim3(1, 4096), nPlanes - o>>>(oF + o, b + o, nPlanes,
							 nActive);
  }
}

template <typename T>
__global__ void Convolution_bp_bias_(T *d_oF, T *d_b, Int nPlanes, Int nActive) {
  Int n = blockIdx.x * 32 + threadIdx.x;
  d_oF+=n;
  TACC t = 0;
  for (Int row = blockIdx.y; row < nActive; row += gridDim.y)
    t += d_oF[row * nPlanes ];
  atomicAdd(&d_b[n], t);
}
template <typename T>
void Convolution_bp_bias(T *d_oF, T *d_b, Int nPlanes, Int nActive) {
  if (nPlanes / 32 > 0)
    Convolution_bp_bias_<<<dim3(nPlanes / 32, 32), 32>>>(d_oF, d_b, nPlanes, nActive);
  if (nPlanes % 32 > 0) {
    Int o = nPlanes / 32 * 32;
    Convolution_bp_bias_<<<dim3(1, 32), nPlanes - o>>>(d_oF + o, d_b + o, nPlanes, nActive);
  }
}

template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_forwardA(T *inFeatures, T *outFeatures, T *w, Int *rules,
			    Int nHot, Int input_nPlanes, Int input_stride,
			    Int output_nPlanes, Int output_stride) {
  // nHot must be a multiple of K!!

  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  Int M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  Int n = blockIdx.y;
  outFeatures += n * K;
  w += n * K;

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	R0[v] = rules[2 * (s + ty[v])];
	R1[v] = rules[2 * (s + ty[v]) + 1];
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  O[v] += I[ty[v]][k] * W[k][tx];

#pragma unroll
      for (int v = 0; v < V; v++)
	O[v] += outFeatures[R1[v] * output_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	outFeatures[R1[v] * output_stride + tx] = O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_forwardB(T *inFeatures, T *outFeatures, T *w, Int *rules,
			    Int nHot, Int input_nPlanes, Int input_stride,
			    Int output_nPlanes, Int output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  Int M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  Int n = blockIdx.y;
  outFeatures += n * K;
  w += n * K;

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (s + ty[v] < nHot) {
	  R0[v] = rules[2 * (s + ty[v])];
	  R1[v] = rules[2 * (s + ty[v]) + 1];
	}
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (s + ty[v] < nHot)
	  I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  O[v] += I[ty[v]][k] * W[k][tx];

#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  O[v] += outFeatures[R1[v] * output_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  outFeatures[R1[v] * output_stride + tx] = O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      Int o = (nHot / K) * K;                                                  \
      if (o >= K)                                                              \
	dConvolution_KMxKN_forwardA<                                           \
	    T, K, V><<<dim3(std::min(o / K, (Int)512), output_nPlanes / K),    \
		       dim3(K, K / V)>>>(inFeatures, outFeatures, w, rules, o, \
					 input_nPlanes, input_stride,          \
					 output_nPlanes, output_stride);       \
      if (nHot > o)                                                            \
	dConvolution_KMxKN_forwardB<                                           \
	    T, K, V><<<dim3(1, output_nPlanes / K), dim3(K, K / V)>>>(         \
	    inFeatures, outFeatures, w, rules + 2 * o, nHot - o,               \
	    input_nPlanes, input_stride, output_nPlanes, output_stride);       \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dConvolution_forward(T *inFeatures, T *outFeatures, T *w, Int *rules,
			  Int nHot, Int input_nPlanes, Int input_stride,
			  Int output_nPlanes, Int output_stride) {
  FOO(T, 64, 16)
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
template <>
void dConvolution_forward<double>(double *inFeatures, double *outFeatures,
				  double *w, Int *rules, Int nHot,
				  Int input_nPlanes, Int input_stride,
				  Int output_nPlanes, Int output_stride) {
  FOO(double, 32, 8)
  FOO(double, 16, 4)
  FOO(double, 8, 2)
  assert(false);
}
#undef FOO

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_backward_dW_A(T *inFeatures, T *dInFeatures, T *dOutFeatures,
				 T *w, T *dw, Int *rules, Int nHot,
				 Int input_nPlanes, Int input_stride,
				 Int output_nPlanes, Int output_stride) {
  // M = gridDim.y == input_nPlanes / K
  Int N = output_nPlanes / K;
  Int m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  w += m * K * output_nPlanes;
  dw += m * K * output_nPlanes;

  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	R0[v] = rules[2 * (s + ty[v])];
	R1[v] = rules[2 * (s + ty[v]) + 1];
	dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++) {
	I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	dO[ty[v]][tx] = dOutFeatures[R1[v] * output_stride + tx];
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++) {
	  dI[v] += dO[ty[v]][k] * W[tx][k];
	  dW[v] += I[k][ty[v]] * dO[k][tx];
	}
#pragma unroll
      for (int v = 0; v < V; v++)
	dI[v] += dInFeatures[R0[v] * input_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	dInFeatures[R0[v] * input_stride + tx] = dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_backward_dW_B(T *inFeatures, T *dInFeatures, T *dOutFeatures,
				 T *w, T *dw, Int *rules, Int nHot,
				 Int input_nPlanes, Int input_stride,
				 Int output_nPlanes, Int output_stride) {
  // M = gridDim.y == input_nPlanes / K
  Int N = output_nPlanes / K;
  Int m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  w += m * K * output_nPlanes;
  dw += m * K * output_nPlanes;

  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (s + ty[v] < nHot) {
	  R0[v] = rules[2 * (s + ty[v])];
	  R1[v] = rules[2 * (s + ty[v]) + 1];
	}
	dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot) {
	  I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	  dO[ty[v]][tx] = dOutFeatures[R1[v] * output_stride + tx];
	} else {
	  I[ty[v]][tx] = 0;
	  dO[ty[v]][tx] = 0;
	}
      __syncthreads();
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++) {
	  dI[v] += dO[ty[v]][k] * W[tx][k];
	  dW[v] += I[k][ty[v]] * dO[k][tx];
	}
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  dI[v] += dInFeatures[R0[v] * input_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  dInFeatures[R0[v] * input_stride + tx] = dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      Int o = (nHot / K) * K;                                                  \
      if (o >= K)                                                              \
	dConvolution_KMxKN_backward_dW_A<                                      \
	    T, K, V><<<dim3(std::min(o / K, (Int)512), input_nPlanes / K),     \
		       dim3(K, K / V)>>>(                                      \
	    inFeatures, dInFeatures, dOutFeatures, w, dw, rules, o,            \
	    input_nPlanes, input_stride, output_nPlanes, output_stride);       \
      if (nHot > o)                                                            \
	dConvolution_KMxKN_backward_dW_B<                                      \
	    T, K, V><<<dim3(1, input_nPlanes / K), dim3(K, K / V)>>>(          \
	    inFeatures, dInFeatures, dOutFeatures, w, dw, rules + 2 * o,       \
	    nHot - o, input_nPlanes, input_stride, output_nPlanes,             \
	    output_stride);                                                    \
      return;                                                                  \
    }                                                                          \
  }



// compute dInFeatures, dw based on given dOutFeatures
// only compute dInFeatures currently, however, need to update the input rule
template <typename T, Int K, Int ConvSize>
__global__ void
dConvolution_KMxKN_backward_dI_ChunkBased(T *inFeatures, T *w, unsigned short BY,
                                       InputAddress *input_address, Int nActiveChunks,
                                       T *dInFeatures, T *dOutFeatures, T *dw,
                                       Int input_nPlanes, Int input_stride,
                                       Int output_nPlanes, Int output_stride) {
    __shared__ T local_memory[MAX_INPUT_ADDRESS * K];
    __shared__ T local_weight[ConvSize*16*16];
    __shared__ Int *aI;
    __shared__ short *inputRuleBook;
    __shared__ unsigned short cntInput;
    __shared__ Int *aO;
    __shared__ unsigned short cntOutput;
    for(unsigned short b_y = 0; b_y < BY; b_y++)
    {
        //load weight to shared memory
        for(int i = 0; i < ConvSize; i+=blockDim.z)
        {
            if(i+threadIdx.z < ConvSize)
            {
                local_weight[(i+threadIdx.z) * K * K + (threadIdx.x) * K + threadIdx.y] = w[(i + threadIdx.z)*input_nPlanes*output_nPlanes +
                            (threadIdx.y + b_y * K)* output_nPlanes + (threadIdx.x + blockIdx.x * K ) ];
            }
        }
//        __syncthreads(); // !!! could be merged with the next sync.]

        // each block processes 1 chunk at one time, maybe 100 chunk in total for 1 block
        // 28 times block sizes, maybe 140 blocks run in parallel
        // more points in each block, less synchronize
        for(Int i = 0; i < nActiveChunks; i+= gridDim.z)
        {
            if(i+blockIdx.z >= nActiveChunks)
            {
                continue;
            }
            // used shared memory instead of local structure for better memory usage!
            if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                cntOutput = input_address[i+blockIdx.z].cO;
                cntInput  = input_address[i+blockIdx.z].cI;
                aO = (Int *)input_address[i+blockIdx.z].aO;
                inputRuleBook = (short *)input_address[i + blockIdx.z].inputRuleBook;
                aI = (Int *)input_address[i+blockIdx.z].aI;
            }
            __syncthreads();
            // load global feature memory to shared memory
            for(unsigned short  k = 0; k < cntOutput; k+= blockDim.y*blockDim.z )
            {
                int local_k = k + threadIdx.y + threadIdx.z * blockDim.y;
                if(local_k >= cntOutput) continue;
                local_memory[local_k * K + threadIdx.x] = dOutFeatures[aO[local_k] * output_stride + blockIdx.x * K + threadIdx.x];
            }
            __syncthreads();
            // only block-wise sync, should be efficient enough
            // convolution here, results are saved in shared memory
            // simplest implementation, which could be further improved!
            // 100 outputs approximately, could be further improved
            // blockDim.z outputs are processed at the same time
            for(unsigned short  k = 0; k < cntInput ; k+=blockDim.z * blockDim.y)
            {
                unsigned short  inputIndex = k + threadIdx.z * blockDim.y + threadIdx.y;
                if(inputIndex >= cntInput) continue;
                float sum = 0;
                for(unsigned short  ruleIndex = 0; ruleIndex < ConvSize; ruleIndex++)
                {
                    int outputFeatureIndex = (int)inputRuleBook[ruleIndex + inputIndex * ConvSize];
                    if(outputFeatureIndex < 0) continue;
                    int weight = ruleIndex * K * K + threadIdx.x;
                    for(int n = 0; n < 16; n++)
                    {
                        sum += local_weight[weight + n * K] * local_memory[outputFeatureIndex * K + n];
#if 0
                        atomicAdd(&dw[(ruleIndex)*input_nPlanes*output_nPlanes +
                                (n + b_y * K)* output_nPlanes + (threadIdx.x + blockIdx.x * K ) ],
                                inFeatures[inputFeatureAddress  * input_stride + b_y * K + n] *
                                                          local_memory[outputFeatureIndex*K + threadIdx.x]);
#endif
                    }
                }

                atomicAdd(&dInFeatures[aI[inputIndex]  * input_stride + b_y * K + threadIdx.x],sum);
            }
            __syncthreads();    // only block-wise sync, should be efficient enough
        }
    }
}
#if 0
// compute dInFeatures, dw based on given dOutFeatures
// only compute dInFeatures currently, however, need to update the input rule
template <typename T, Int K, Int ConvSize>
__global__ void
dConvolution_KMxKN_backward_dW_ChunkBased(T *inFeatures, T *w, unsigned short BY,
                                       InputAddress *input_address, Int nActiveChunks,
                                       T *dInFeatures, T *dOutFeatures, T *dw,
                                       Int input_nPlanes, Int input_stride,
                                       Int output_nPlanes, Int output_stride) {
    __shared__ T local_memory[MAX_INPUT_ADDRESS * K];
    __shared__ T local_input[MAX_INPUT_ADDRESS * K];
//    __shared__ T local_input[4*K];
//    __shared__ T local_weight[ConvSize*16*16];
    __shared__ Int *aI;
    __shared__ Int *aO;
    __shared__ unsigned int *inputRuleBook;
    __shared__ unsigned int *outputRuleBook;
    __shared__ unsigned short cntInput;
    __shared__ unsigned short cntOutput;
    for(unsigned short b_y = 0; b_y < BY; b_y++)
    {
#if 0
        //load weight to shared memory
        for(int i = 0; i < ConvSize; i+=blockDim.z)
        {
            if(i+threadIdx.z < ConvSize)
            {

                local_weight[(i+threadIdx.z) * K * K + (threadIdx.y) * K + threadIdx.x] = 0;
            }
        }
#endif
//        __syncthreads(); // !!! could be merged with the next sync.]

        // each block processes 1 chunk at one time, maybe 100 chunk in total for 1 block
        // 28 times block sizes, maybe 140 blocks run in parallel
        // more points in each block, less synchronize
        for(unsigned short i = 0; i < nActiveChunks; i+= gridDim.z)
        {
            if(i+blockIdx.z >= nActiveChunks)
            {
                continue;
            }

            // used shared memory instead of local structure for better memory usage!
//            if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                inputRuleBook = (unsigned int *)input_address[i + blockIdx.z].inputRuleBook;
                outputRuleBook = (unsigned int *)input_address[i+blockIdx.z].outputRuleBook;
                cntInput = input_address[i+blockIdx.z].cI;
                cntOutput = input_address[i+blockIdx.z].cO;
                aI = (Int *)input_address[i+blockIdx.z].aI;
                aO = (Int *)input_address[i+blockIdx.z].aO;
            }
//            __syncthreads();
            // load global feature memory to shared memory
            for(unsigned short  k = 0; k < cntOutput; k+= blockDim.y*blockDim.z )
            {
                int local_k = k + threadIdx.y + threadIdx.z * blockDim.y;
                if(local_k >= cntOutput) continue;
                local_memory[local_k * K + threadIdx.x] = dOutFeatures[aO[local_k] * output_stride + blockIdx.x * K + threadIdx.x];
            }
            for(unsigned short  k = 0; k < cntOutput; k+= blockDim.y*blockDim.z )
            {
                int local_k = k + threadIdx.y + threadIdx.z * blockDim.y;
                if(local_k >= cntOutput) continue;
                local_input[local_k * K + threadIdx.x] = inFeatures[aI[local_k] * input_stride + b_y * K + threadIdx.x];
            }
            __syncthreads();
            // only block-wise sync, should be efficient enough
            // convolution here, results are saved in shared memory
            // simplest implementation, which could be further improved!
            // 100 outputs approximately, could be further improved
            // blockDim.z outputs are processed at the same time
            for(unsigned short  k = 0; k < cntInput ; k+=blockDim.z)
            {
                unsigned short  inputIndex = k + threadIdx.z;
                if(inputIndex >= cntInput) continue;
#if 0
                if(threadIdx.y == 0 )
                {
                    local_input[threadIdx.z * K + threadIdx.x] = inFeatures[aI[inputIndex] * input_stride + b_y * K + threadIdx.x];
                }
                __syncthreads();    // only block-wise sync, should be efficient enough
#endif
                for(unsigned short  ruleIndex = 0; ruleIndex < ConvSize; ruleIndex++)
                {
                    Int outputFeatureIndex = inputRuleBook[ruleIndex + inputIndex * ConvSize];
                    if(outputFeatureIndex < 0) continue;
#if 1
                      atomicAdd(&dw[(ruleIndex)*input_nPlanes*output_nPlanes +
                              (threadIdx.y + b_y * K)* output_nPlanes + (threadIdx.x + blockIdx.x * K ) ],
//                    atomicAdd(&local_weight[ruleIndex * K * K + threadIdx.y*K + threadIdx.x],
                            local_input[inputIndex * K + threadIdx.y] *
//                        inFeatures[aI[inputIndex] * input_stride + b_y * K + threadIdx.y] *
                                                        local_memory[outputFeatureIndex*K + threadIdx.x]);
#endif
                }
            }
            __syncthreads();    // only block-wise sync, should be efficient enough
        }
#if 0
        for(int i = 0; i < ConvSize; i+=blockDim.z)
        {
            if(i+threadIdx.z < ConvSize)
            {
                atomicAdd(&dw[(i + threadIdx.z)*input_nPlanes*output_nPlanes +
                        (threadIdx.y + b_y * K)* output_nPlanes + (threadIdx.x + blockIdx.x * K ) ],
                        local_weight[(i+threadIdx.z) * K * K + (threadIdx.y) * K + threadIdx.x] );
            }
        }
#endif
    }
}
#else


// rulebook based convolution, 27 rule book, one for each

//divide and compute?
//should support 32x32 as well
// bx: gridDim.x, input
// by: gridDim.y, output
#if 0
template <typename T, Int K, Int ConvSize>
__global__ void
dConvolution_KMxKN_backward_dW_RuleBookBased(T *inFeatures, Int * rulebook,Int nRules,
                                             T *dOutFeatures, T *dw, Int input_stride,
                                       Int output_nPlanes, Int output_stride) {
    __shared__ T local_dw[K*K];
    __shared__ T local_input[4 * K * K];
    __shared__ T local_output[4 * K * K];
    if(threadIdx.z == 0)
    {
        local_dw[(threadIdx.y) * K + threadIdx.x] = 0;
    }
    for(Int i = 0; i < nRules; i += gridDim.z * blockDim.y * blockDim.z)
    {
        Int local_index = i + threadIdx.y  + blockIdx.z * blockDim.y * blockDim.z + threadIdx.z * blockDim.y;
        __syncthreads();
        if(local_index < nRules)
        {
            local_input[threadIdx.z * blockDim.y * K + threadIdx.y * K + threadIdx.x] = inFeatures[rulebook[(local_index ) *2] * input_stride + blockIdx.x * K + threadIdx.x];
            local_output[threadIdx.z * blockDim.y * K + threadIdx.y * K + threadIdx.x] = dOutFeatures[rulebook[(local_index) * 2 + 1] * output_stride + blockIdx.y * K + threadIdx.x];
        }
        __syncthreads();
        for(int N = 0; N < 16 ; N ++)
        {
            if(i+N+blockIdx.z * 16<nRules)
            atomicAdd(&local_dw[threadIdx.y * K + threadIdx.x],
                    local_input[threadIdx.z * 16 * K + N * K + threadIdx.y] * local_output[threadIdx.z * 16 * K + N * K + threadIdx.x]);
        }

    }
    atomicAdd(&dw[ (threadIdx.y + blockIdx.x * K)* output_nPlanes + (threadIdx.x + blockIdx.y * K )] ,
            local_dw[(threadIdx.y) * K + threadIdx.x]) ;

}
#else
template <typename T, Int K, Int ConvSize>
__global__ void
dConvolution_KMxKN_backward_dW_RuleBookBased(T *inFeatures, Int * rulebook,Int nRules,
                                             T *dOutFeatures, T *dw, Int input_stride,
                                       Int output_nPlanes, Int output_stride) {
    __shared__ T local_input[K * K];
    __shared__ T local_output[K * K];
    T local_dw = 0;
    for(Int i = 0; i <  nRules - nRules % (gridDim.z * blockDim.y); i += gridDim.z * blockDim.y)
    {
        Int local_index = i + threadIdx.y  + blockIdx.z * blockDim.y;
        __syncthreads();
        local_input[threadIdx.y * K + threadIdx.x] = inFeatures[rulebook[(local_index ) *2] * input_stride + blockIdx.x * K + threadIdx.x];
        local_output[threadIdx.y * K + threadIdx.x] = dOutFeatures[rulebook[(local_index) * 2 + 1] * output_stride + blockIdx.y * K + threadIdx.x];
        __syncthreads();

#if 1
        for(int N = 0; N < 16 ; N ++)
        {
            local_dw += local_input[N * K + threadIdx.y] * local_output[N * K + threadIdx.x];
        }
#else
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[0 * K + threadIdx.y] ,local_output[0 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[1 * K + threadIdx.y] ,local_output[1 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[2 * K + threadIdx.y] ,local_output[2 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[3 * K + threadIdx.y] ,local_output[3 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[4 * K + threadIdx.y] ,local_output[4 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[5 * K + threadIdx.y] ,local_output[5 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[6 * K + threadIdx.y] ,local_output[6 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[7 * K + threadIdx.y] ,local_output[7 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[8 * K + threadIdx.y] ,local_output[8 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[9 * K + threadIdx.y] ,local_output[9 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[10 * K + threadIdx.y] ,local_output[10 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[11 * K + threadIdx.y] ,local_output[11 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[12 * K + threadIdx.y] ,local_output[12 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[13 * K + threadIdx.y] ,local_output[13 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[14 * K + threadIdx.y] ,local_output[14 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
        local_dw[threadIdx.y * K + threadIdx.x] = __fmaf_rn(local_input[15 * K + threadIdx.y] ,local_output[15 * K + threadIdx.x],local_dw[threadIdx.y * K + threadIdx.x]);
#endif
    }
    Int local_index = nRules - nRules % (gridDim.z * blockDim.y) + threadIdx.y  + blockIdx.z * blockDim.y;

    __syncthreads();
    if(local_index < nRules)
    {
        local_input[threadIdx.y * K + threadIdx.x] = inFeatures[rulebook[(local_index ) *2] * input_stride + blockIdx.x * K + threadIdx.x];
        local_output[threadIdx.y * K + threadIdx.x] = dOutFeatures[rulebook[(local_index) * 2 + 1] * output_stride + blockIdx.y * K + threadIdx.x];
    }
    __syncthreads();
    for(int N = 0; N < 16 ; N ++)
    {
        if(local_index - threadIdx.y + N <nRules)
        local_dw += local_input[N * K + threadIdx.y] * local_output[N * K + threadIdx.x];
    }
    atomicAdd(&dw[ (threadIdx.y + blockIdx.x * K)* output_nPlanes + (threadIdx.x + blockIdx.y * K )] ,
            local_dw) ;


}
#endif

#endif

template <typename T>
void dConvolution_backward_chunkbased(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                  T *w, T *dw, void *input_address, Int nActiveChunks,
                  Int input_nPlanes, Int input_stride,
                  Int output_nPlanes, Int output_stride, RuleBook &rules) {
    int bX = output_nPlanes / 16;
    int bY = 1;
    int bZ = max(1536/ (bX * bY),1);


#if DEBUG_FAST_BACKWARD
    clock_t start,end;
    double time_dI,time_dw;
    cudaDeviceSynchronize();
    start = clock();
    cudaDeviceSynchronize();
#endif
#if 1
//    printf("nActive chunks: %d\r\n", nActiveChunks);
    dConvolution_KMxKN_backward_dI_ChunkBased<T,16,27><<<dim3(bX,bY,bZ), dim3(16,16,4)>>>
        (inFeatures, w, input_nPlanes / 16,
         (InputAddress *)input_address, nActiveChunks, dInFeatures, dOutFeatures, dw,
         input_nPlanes, input_stride,
         output_nPlanes, output_stride);
#endif
#if DEBUG_FAST_BACKWARD
    cudaDeviceSynchronize();
    end = clock();
    time_dI = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
    start = clock();
#endif
    Int rbMaxSize = 0;
    for (auto &r : rules)
      rbMaxSize = std::max(rbMaxSize, (Int)r.size());
    Int *rbB;
    HANDLE_ERROR(cudaMalloc( (void**)&rbB, rbMaxSize * sizeof(Int)));
    for(int i = 0; i < rules.size(); i++)
    {
        auto &r = rules[i];
        Int nHotB = r.size() / 2;
        HANDLE_ERROR(cudaMemcpy(rbB, &r[0], sizeof(Int) * 2 * nHotB, cudaMemcpyHostToDevice));
//        printf("nrules: %d\r\n", nHotB);
#if 1
        (dConvolution_KMxKN_backward_dW_RuleBookBased<T,16,27><<<dim3(input_nPlanes / 16,output_nPlanes / 16,1024), dim3(16,16,1)>>>
                    (inFeatures, rbB, nHotB, dOutFeatures, dw, input_stride,
                     output_nPlanes, output_stride));
#endif
        dw += input_nPlanes * output_nPlanes;
    }
    HANDLE_ERROR(cudaFree(rbB));
#if DEBUG_FAST_BACKWARD
    cudaDeviceSynchronize();
    end = clock();
    time_dw = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
    printf("%d %d %f %f\r\n", input_nPlanes, output_nPlanes, time_dI, time_dw);
#endif
}
template <typename T>
void dConvolution_backward_dW(T *inFeatures, T *dInFeatures, T *dOutFeatures,
			      T *w, T *dw, Int *rules, Int nHot,
			      Int input_nPlanes, Int input_stride,
			      Int output_nPlanes, Int output_stride) {
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
#undef FOO

template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_forward2(T *inFeatures, T *outFeatures, T *w, Int *rules,
			    Int nHot, Int input_nPlanes, Int input_stride,
			    Int output_nPlanes, Int output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nHot x input_nplanes<=KM -> nHot x output_nPlanes<=KN
  // - parallel over N,nHot - loop over M

  Int M = (input_nPlanes + K - 1) / K;
  // N = gridDim.y ~ output_nPlanes/K
  Int n = blockIdx.y;
  outFeatures += n * K;
  w += n * K;
  Int KO = min(K, output_nPlanes - K * n);

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  __shared__ Int R[K * 2];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
    Int KI = min(K, input_nPlanes - K * m);

// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
	      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
// Read rules for K input/output pairs
#pragma unroll
      for (int v = 0; v < V; v++) {
        if (ty[v] < 2) {
          int q = ty[v] * K + tx;
          if (s + q / 2 < nHot)
            R[q] = rules[2 * s + q];
        }
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
        if (tx < KI and s + ty[v] < nHot)
          I[ty[v]][tx] = inFeatures[R[2 * ty[v]] * input_stride + tx];
        O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < KI; k++)
#pragma unroll
        for (int v = 0; v < V; v++)
            O[v] += I[ty[v]][k] * W[k][tx];
      __syncthreads();

#pragma unroll
      for (int v = 0; v < V; v++)
        if (tx < KO and s + ty[v] < nHot)
          outFeatures[R[2 * ty[v] + 1] * output_stride + tx] += O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
				T *w, T *dw, Int *rules, Int nHot,
				Int input_nPlanes, Int input_stride,
				Int output_nPlanes, Int output_stride) {
  // M = gridDim.y == input_nPlanes / K
  Int N = (output_nPlanes + K - 1) / K;
  Int m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  w += m * K * output_nPlanes;
  dw += m * K * output_nPlanes;
  Int KI = min(K, input_nPlanes - K * m);

  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  __shared__ Int R[K * 2];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
    Int KO = min(K, output_nPlanes - K * n);

// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      if (ty[v] < KI and tx < KO)
	W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
// Read rules for K input/output pairs, reset dI[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (ty[v] < 2) {
	  int q = ty[v] * K + tx;
	  if (s + q / 2 < nHot)
	    R[q] = rules[2 * s + q];
	}
	dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (tx < KI and s + ty[v] < nHot)
	  I[ty[v]][tx] = inFeatures[R[2 * ty[v]] * input_stride + tx];
	else
	  I[ty[v]][tx] = 0;
	if (tx < KO and s + ty[v] < nHot)
	  dO[ty[v]][tx] = dOutFeatures[R[2 * ty[v] + 1] * output_stride + tx];
	else
	  dO[ty[v]][tx] = 0;
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < KO; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  dI[v] += dO[ty[v]][k] * W[tx][k];
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  dW[v] += I[k][ty[v]] * dO[k][tx];
      __syncthreads();
#pragma unroll
      for (int v = 0; v < V; v++)
	if (tx < KI and s + ty[v] < nHot)
	  dInFeatures[R[2 * ty[v]] * input_stride + tx] += dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
	atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}

template <typename T>
double dConvolution_forward2(T *inFeatures, T *outFeatures, T *w,
			     RuleBook _rules, Int input_nPlanes,
			     Int input_stride, Int output_nPlanes,
			     Int output_stride) {

  EASY_FUNCTION(profiler::colors::Cyan);

  Int c = input_nPlanes * output_nPlanes;
  double flops = 0;
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    RULEBOOKITERATOR(
	(dConvolution_KMxKN_forward2<
	    T, K,
	    V><<<dim3(128, (output_nPlanes + K - 1) / K), dim3(K, K / V)>>>(
	    inFeatures, outFeatures, w, rbB, nHotB, input_nPlanes, input_stride,
	    output_nPlanes, output_stride));
	, w += c; flops += nHotB * c;)
  } else {
    RULEBOOKITERATOR(dConvolution_forward(inFeatures, outFeatures, w, rbB,
					  nHotB, input_nPlanes, input_stride,
					  output_nPlanes, output_stride);
		     , w += c; flops += nHotB * c;)
  }
  return flops;
}

template <typename T>
void dConvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
			       T *w, T *dw, RuleBook _rules, Int input_nPlanes,
			       Int input_stride, Int output_nPlanes,
			       Int output_stride) {

  EASY_FUNCTION(profiler::colors::Cyan100);

  Int c = input_nPlanes * output_nPlanes;
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    RULEBOOKITERATOR(
	(dConvolution_KMxKN_backward_dW2<
	    T, K,
	    V><<<dim3(128, (input_nPlanes + K - 1) / K), dim3(K, K / V)>>>(
	    inFeatures, dInFeatures, dOutFeatures, w, dw, rbB, nHotB,
	    input_nPlanes, input_stride, output_nPlanes, output_stride));
	, w += c; dw += c;)
  } else {
    RULEBOOKITERATOR(dConvolution_backward_dW(inFeatures, dInFeatures,
					      dOutFeatures, w, dw, rbB, nHotB,
					      input_nPlanes, input_stride,
					      output_nPlanes, output_stride);
		     , w += c; dw += c;)
  }
}

/* Chunk based*/


#if 1
// blockIdx.x, blockIdx.y, blockIdx.z
// K = 16, ConvSize = 27
// griddim = (outputplanes/16, 1, 192/(outputplanes/16))
// blockdim = (16, 16, 4)
// parallel over every output point, for every output point, 

template <typename T, Int K, Int ConvSize>
__global__ void
dConvolution_KMxKN_forwardA_ChunkBased(T *inFeatures/* NActive*inputplanes */, T *outFeatures /* NActive*outputplanes */, 
                                       T *w /* ConvSize*inputplane*outputplane */, Int BY /*inputplane/16*/,
                                       InputAddress *input_address, Int nActiveChunks /* chunk num */, Int input_nPlanes, Int input_stride,
                                       Int output_nPlanes, Int output_stride) {
    __shared__ T local_memory[MAX_INPUT_ADDRESS * K];
    __shared__ T local_weight[ConvSize*K*K];

    for(Int b_y = 0; b_y < BY ; b_y++)
    {
        //load weight to shared memory
        for(Int i = 0; i < ConvSize; i+=blockDim.z)
        {
            if(i+threadIdx.z < ConvSize)
            {
                // one block load convsize*K*K weights out of convsize*inp*oup weights
                local_weight[(i+threadIdx.z) * K * K + (threadIdx.y) * K + threadIdx.x] = w[(i + threadIdx.z)*input_nPlanes*output_nPlanes +
                            (threadIdx.y + b_y * K)* output_nPlanes + (threadIdx.x + blockIdx.x * K ) ];
            }
        }
//        __syncthreads(); // !!! could be merged with the next sync.]

        // each block processes 1 chunk at one time, maybe 100 chunk in total for 1 block
        // 28 times block sizes, maybe 140 blocks run in parallel
        // more points in each block, less synchronize
        for(Int i = 0; i < nActiveChunks; i+= gridDim.z)
        {
            if(i+blockIdx.z >= nActiveChunks)
            {
                continue;
            }
            Int *aI;
            Int *aO;
            short cntInput;
            short cntOutput;
            short *aR;
            // used shared memory to hide latency!
           // if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            {
                aI = (Int *)input_address[i+blockIdx.z].aI;
                aO = (Int *)input_address[i+blockIdx.z].aO;
                cntInput = input_address[i+blockIdx.z].cI;
                cntOutput = input_address[i+blockIdx.z].cO;
                aR = (short *)input_address[i+blockIdx.z].outputRuleBook;
            }

            for(Int k = 0; k < cntInput; k+= blockDim.y*blockDim.z )
            {
                int local_k = k + threadIdx.y + threadIdx.z * blockDim.y;
                if(local_k >= cntInput) continue;
                // load (cntInput * K) weights, b_y controls the start column.
                local_memory[local_k * K + threadIdx.x] = inFeatures[aI[local_k] * input_stride + b_y * K + threadIdx.x];
            }
            __syncthreads();    // only block-wise sync, should be efficient enough
            // convolution here, results are saved in shared memory
            // simplest implementation, which could be further improved!
            // 100 outputs approximately, could be further improved
            // blockDim.z outputs are processed at the same time

            for(short k = 0; k < cntOutput; k+=blockDim.z * blockDim.y)
            {
                Int outputIndex = k + threadIdx.z * blockDim.y + threadIdx.y;
                if(outputIndex >= cntOutput) continue;

                T sum = 0;
                #pragma unroll
                for(short ruleIndex = 0; ruleIndex < ConvSize; ruleIndex++)
                {   
                  // local input feats start address.
                    int iF = (int)aR[outputIndex * ConvSize + ruleIndex]*K;//local_aR[localIndex + ruleIndex - ruleStart] * K;
                    if(iF < 0)
                    {
                        continue;
                    }
                    int weight = ruleIndex * K * K + threadIdx.x;//local_aR[localIndex + ruleIndex - ruleStart + MAXIMUM_RULES] * K * K + 0;
                    #pragma unroll
                    for (int _ = 0; _ < K; _++) {
                      // sum = local_weight[weight] * local_memory[iF] + sum
                      sum = __fmaf_rn(local_weight[weight],local_memory[iF],sum);
                      // input * weight = output 
                      weight += K; 
                      iF++;
                    }
                }
                atomicAdd(&outFeatures[aO[outputIndex] * output_stride + blockIdx.x * K + threadIdx.x],sum);
            }
            __syncthreads();    // only block-wise sync, should be efficient enough
        }
    }
}
#else
// blockIdx.x, blockIdx.y, blockIdx.z
#define MAXIMUM_RULES (16*4*32)
template <typename T, Int K, Int ConvSize>
__global__ void
dConvolution_KMxKN_forwardA_ChunkBased(T *inFeatures, T *outFeatures, T *w, Int BY,
                                       InputAddress *input_address, Int nActiveChunks, Int input_nPlanes, Int input_stride,
                                       Int output_nPlanes, Int output_stride) {
    __shared__ T local_memory[MAX_INPUT_ADDRESS * K];
    __shared__ T local_weight[27*16*16];

    Int *aI;
    Int *aO;
    Int cntInput;
    Int cntOutput;
    unsigned int *aR;
    for(Int b_y = 0; b_y < BY; b_y++)
    {
        //load weight to shared memory
        for(Int i = 0; i < ConvSize; i+=blockDim.z)
        {
            if(i+threadIdx.z < ConvSize)
            {
                local_weight[(i+threadIdx.z) * K * K + (threadIdx.y) * K + threadIdx.x] = w[(i + threadIdx.z)*input_nPlanes*output_nPlanes +
                            (threadIdx.y + b_y * K)* output_nPlanes + (threadIdx.x + blockIdx.x * K ) ];
            }
        }
//        __syncthreads(); // !!! could be merged with the next sync.]

        // each block processes 1 chunk at one time, maybe 100 chunk in total for 1 block
        // 28 times block sizes, maybe 140 blocks run in parallel
        // more points in each block, less synchronize
        for(Int i = 0; i < nActiveChunks; i+= gridDim.z)
        {
            if(i+blockIdx.z >= nActiveChunks)
            {
                continue;
            }
            // used shared memory to hide latency!

                aI = (Int *)input_address[i+blockIdx.z].aI;
                aO = (Int *)input_address[i+blockIdx.z].aO;
                cntInput = input_address[i+blockIdx.z].cI;
                cntOutput = input_address[i+blockIdx.z].cO;
                aR = (unsigned int *)input_address[i+blockIdx.z].inputRuleBook;

            for(Int  k = 0; k < cntInput; k+= blockDim.y*blockDim.z )
            {
                int local_k = k + threadIdx.y + threadIdx.z * blockDim.y;
                if(local_k >= cntInput) continue;
                local_memory[local_k * K + threadIdx.x] = inFeatures[aI[local_k] * input_stride + b_y * K + threadIdx.x];
            }
            __syncthreads();    // only block-wise sync, should be efficient enough
            // convolution here, results are saved in shared memory
            // simplest implementation, which could be further improved!
            // 100 outputs approximately, could be further improved
            // blockDim.z outputs are processed at the same time

            for(Int  k = 0; k < cntOutput; k+=blockDim.z * blockDim.y)
            {
                Int  outputIndex = k + threadIdx.z * blockDim.y + threadIdx.y;
                if(outputIndex >= cntOutput) continue;

                T sum = 0;
                T data;
                for(Int  ruleIndex = 0; ruleIndex < 27; ruleIndex++)
                {
                    int iF = aR[outputIndex * 27 + ruleIndex]*K;//local_aR[localIndex + ruleIndex - ruleStart] * K;
                    if(iF < 0)
                    {
                        continue;
                    }
                    int weight = ruleIndex * K * K;//local_aR[localIndex + ruleIndex - ruleStart + MAXIMUM_RULES] * K * K + 0;

                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                    data = local_weight[weight+threadIdx.x]; sum = __fmaf_rn(data,local_memory[iF],sum); weight += 16; iF++;
                }
                atomicAdd(&outFeatures[aO[outputIndex] * output_stride + blockIdx.x * K + threadIdx.x],sum);

            }
            __syncthreads();    // only block-wise sync, should be efficient enough
        }
    }
}
#endif
// convolution here, input should be configured. at least build a test case for this environment.
template <typename T>
void dConvolution_forward_ChunkBased(T *inFeatures, T *outFeatures, T *w,
                                     void *input_address,  Int nActiveChunks,
                                     Int input_nPlanes, Int input_stride,
                                     Int output_nPlanes, Int output_stride)
{
    EASY_FUNCTION(profiler::colors::Magenta);

    int bX = output_nPlanes / 16;
    int bY = 1;
    int bZ = max(192 / (bX * bY),1);
    dConvolution_KMxKN_forwardA_ChunkBased<T,16,27><<<dim3(bX,bY,bZ), dim3(16,16,4)>>>(inFeatures, outFeatures, w, input_nPlanes / 16,
                                           (InputAddress *)input_address, nActiveChunks, input_nPlanes, input_stride,
                                           output_nPlanes, output_stride);

}

template <typename T>
void dConvolution_incre_forward_ChunkBased(T *inFeatures, T *outFeatures, T *w,
                                     void *input_address,  Int nActiveChunks,
                                     Int input_nPlanes, Int input_stride,
                                     Int output_nPlanes, Int output_stride)
{
    EASY_FUNCTION(profiler::colors::Magenta);
    
    int bX = output_nPlanes / 16;
    int bY = 1;
    int bZ = max(192 / (bX * bY),1);
    dConvolution_KMxKN_forwardA_ChunkBased<T,16,27><<<dim3(bX,bY,bZ), dim3(16,16,4)>>>(inFeatures, outFeatures, w, input_nPlanes / 16,
                                           (InputAddress *)input_address, nActiveChunks, input_nPlanes, input_stride,
                                           output_nPlanes, output_stride);
}


// re implement convolution here!
// No need to use RulebookIterator here, and only test simpleast cases here.
// It matters if we can reduce 49ms to 5ms.
template <typename T>
double dConvolution_forward2_chunkbased(T *inFeatures, T *outFeatures, T *w,
                                        Int outFeatureNum, RBChunkPointerList& new_rbChunkList, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes, Int output_stride) {

  EASY_FUNCTION(profiler::colors::Magenta);

  clock_t start,end;
  double time_gt, time_fast, time_prepare_data;

  Int c = input_nPlanes * output_nPlanes;

  double flops = 0;
  // if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
  //   const int K = 16;
  //   const int V = 4;
  //   RULEBOOKITERATOR(
  //   (dConvolution_KMxKN_forward2<T, K, V>
  //       <<<dim3(128, (output_nPlanes + K - 1) / K), dim3(K, K / V)>>>(
  //       inFeatures, outFeatures, w, rbB, nHotB, input_nPlanes, input_stride,
  //       output_nPlanes, output_stride));
  //   , w += c; flops += nHotB * c;)
  // } else {

#if DEBUG_FAST_CONV
    T * weight = w;
    T *initOutputFeaturesDevice;
    cudaMalloc( (void**)&initOutputFeaturesDevice, outFeatureNum * output_nPlanes * sizeof(T) );
    HANDLE_ERROR(cudaMemcpy( initOutputFeaturesDevice, outFeatures, outFeatureNum * output_nPlanes * sizeof(T), cudaMemcpyDeviceToDevice));

    cudaDeviceSynchronize();
    start = clock();
    RULEBOOKITERATOR(dConvolution_forward(inFeatures, initOutputFeaturesDevice, weight , rbB,
                   nHotB, input_nPlanes, input_stride,
                   output_nPlanes, output_stride);
              , weight += c; flops += nHotB * c;)
    cudaDeviceSynchronize();
    end = clock();
    time_gt = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
    T *outputFeatureCPU = new T[outFeatureNum * output_nPlanes];
    HANDLE_ERROR(cudaMemcpy( outputFeatureCPU, initOutputFeaturesDevice, outFeatureNum * output_nPlanes * sizeof(T), cudaMemcpyDeviceToHost));


#endif
#if BUILD_WITH_EASY_PROFILER
    cudaDeviceSynchronize();
#endif

    start = clock();
    Int totalOutputs = 0, validOutputs = 0;
    // these memory could be put into constant memory
    // printf("to here0");
    // InputAddress *inputAddress = new InputAddress[validChunks];
    InputAddress *new_inputAddress = new_rbChunkList.list.data();
    int new_validChunkCnt=new_rbChunkList.list.size();
    void *inputAddress_device;
    HANDLE_ERROR(cudaMalloc( (void**)&inputAddress_device, new_validChunkCnt * sizeof(InputAddress)));
    HANDLE_ERROR(cudaMemcpy( inputAddress_device, new_inputAddress, new_validChunkCnt * sizeof(InputAddress), cudaMemcpyHostToDevice));

// cuda settings
//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    dConvolution_forward_ChunkBased(inFeatures, outFeatures, w,
                                    inputAddress_device, new_validChunkCnt, input_nPlanes, input_stride,
                                    output_nPlanes, output_stride);
#if BUILD_WITH_EASY_PROFILER
    cudaDeviceSynchronize();
#endif


    end = clock();
    time_fast = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;


#if DEBUG_FAST_CONV
    printf("gt/fast: %f %f\r\n", time_gt, time_fast);
    T *outputFeature_test_CPU = new T[outFeatureNum * output_nPlanes];
    HANDLE_ERROR(cudaMemcpy( outputFeature_test_CPU, outFeatures,
                             outFeatureNum * output_nPlanes * sizeof(T), cudaMemcpyDeviceToHost));
    for(int i = 0; i <outFeatureNum * output_nPlanes; i++ )
    {
        if(fabs(outputFeatureCPU[i] -
                outputFeature_test_CPU[i]) >
           fmax(1e-2 * fabs(outputFeatureCPU[i]),1e-4))
        {
            printf("wrong match! %d %f %f\r\n",i,
                   outputFeatureCPU[i],
                    outputFeature_test_CPU[i]);
            exit(0);
            break;
        }
    }
    printf("pass verification!");

    HANDLE_ERROR(cudaFree(initOutputFeaturesDevice));
#endif
    // delete [] inputAddress;
    HANDLE_ERROR(cudaFree(inputAddress_device));
  // }
  return flops;
}


template <typename T>
double dConvolution_incre_forward2_chunkbased(T *inFeatures, T *outFeatures, T *w,
                                        Int outFeatureNum, RBChunkPointerList& new_rbChunkList, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes, Int output_stride) {

  EASY_FUNCTION(profiler::colors::Magenta);


  Int c = input_nPlanes * output_nPlanes;

  double flops = 0;
  //  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
  //   const int K = 16;
  //   const int V = 4;
  //   RULEBOOKITERATOR(
  //   (dConvolution_KMxKN_forward2<T, K, V>
  //       <<<dim3(128, (output_nPlanes + K - 1) / K), dim3(K, K / V)>>>(
  //       inFeatures, outFeatures, w, rbB, nHotB, input_nPlanes, input_stride,
  //       output_nPlanes, output_stride));
  //   , w += c; flops += nHotB * c;)
  //  }
  //   // else {
  //   //   RULEBOOKITERATOR(dConvolution_forward(inFeatures, outFeatures, w, rbB,
  //   //     nHotB, input_nPlanes, input_stride,
  //   //     output_nPlanes, output_stride);
  //   //   , w += c; flops += nHotB * c;)

  //   // }
  //  else {

    // these memory could be put into constant memory
    // InputAddress *inputAddress = new InputAddress[validChunks];
    InputAddress *new_inputAddress = new_rbChunkList.list.data();
    int new_validChunkCnt=new_rbChunkList.list.size();
    void *inputAddress_device;
    HANDLE_ERROR(cudaMalloc( (void**)&inputAddress_device, new_validChunkCnt * sizeof(InputAddress)));
    // why need host to device? InputAddress saved pointer is aready in device?
    HANDLE_ERROR(cudaMemcpy( inputAddress_device, new_inputAddress, new_validChunkCnt * sizeof(InputAddress), cudaMemcpyHostToDevice));

    dConvolution_incre_forward_ChunkBased(inFeatures, outFeatures, w,
                                    inputAddress_device, new_validChunkCnt, input_nPlanes, input_stride,
                                    output_nPlanes, output_stride);



    HANDLE_ERROR(cudaFree(inputAddress_device));
  // }
  return flops;
}

template <typename T>
double dConvolution_incre_forward2(T *inFeatures, T *outFeatures, T *w,
                                        RuleBook &_rules, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes, Int output_stride) {

  EASY_FUNCTION(profiler::colors::Magenta);


  Int c = input_nPlanes * output_nPlanes;

  double flops = 0;
   if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    RULEBOOKITERATOR(
    (dConvolution_KMxKN_forward2<T, K, V>
        <<<dim3(128, (output_nPlanes + K - 1) / K), dim3(K, K / V)>>>(
        inFeatures, outFeatures, w, rbB, nHotB, input_nPlanes, input_stride,
        output_nPlanes, output_stride));
    , w += c; flops += nHotB * c;)
   }
    else {
      RULEBOOKITERATOR(dConvolution_forward(inFeatures, outFeatures, w, rbB,
        nHotB, input_nPlanes, input_stride,
        output_nPlanes, output_stride);
      , w += c; flops += nHotB * c;)

    }

  return flops;
}

template <typename T>
void dConvolution_backward_dW2_chunkbased(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                        RBChunkPointerList& new_rbChunkList,RuleBook &_rules,Int inputFeatureSize,
                                        T *w, T *dw, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes,
                                        Int output_stride)
{
    Int c = input_nPlanes * output_nPlanes;
    if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
      const int K = 16;
      const int V = 4;
      RULEBOOKITERATOR(
      (dConvolution_KMxKN_backward_dW2<
          T, K,
          V><<<dim3(128, (input_nPlanes + K - 1) / K), dim3(K, K / V)>>>(
          inFeatures, dInFeatures, dOutFeatures, w, dw, rbB, nHotB,
          input_nPlanes, input_stride, output_nPlanes, output_stride));
      , w += c; dw += c;)
    } else {
#if 1
    InputAddress *new_inputAddress = new_rbChunkList.list.data();
    int new_validChunkCnt=new_rbChunkList.list.size();

    void *inputAddress_device;
    HANDLE_ERROR(cudaMalloc( (void**)&inputAddress_device, new_validChunkCnt * sizeof(InputAddress)));
    HANDLE_ERROR(cudaMemcpy( inputAddress_device, new_inputAddress, new_validChunkCnt * sizeof(InputAddress), cudaMemcpyHostToDevice));

#endif


#if DEBUG_FAST_BACKWARD
    T *dInFeatures_device;
    T *dw_device;
    T *w_device;
    HANDLE_ERROR(cudaMalloc( (void**)&dInFeatures_device, inputFeatureSize * input_nPlanes * sizeof(T)));
    HANDLE_ERROR(cudaMalloc( (void**)&dw_device, input_nPlanes * output_nPlanes * 27 * sizeof(T)));
    HANDLE_ERROR(cudaMalloc( (void**)&w_device, input_nPlanes * output_nPlanes * 27 * sizeof(T)));
    HANDLE_ERROR(cudaMemcpy(dInFeatures_device, dInFeatures, inputFeatureSize * input_nPlanes * sizeof(T), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(dw_device, dw, input_nPlanes * output_nPlanes * 27 * sizeof(T), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(w_device, w, input_nPlanes * output_nPlanes * 27 * sizeof(T), cudaMemcpyDeviceToDevice));
#endif

#if DEBUG_FAST_BACKWARD
   cudaDeviceSynchronize();

    clock_t start,end;
    double time_gt, time_fast;
    start = clock();
    // try to modify here for better performance!
    RULEBOOKITERATOR(dConvolution_backward_dW(inFeatures, dInFeatures_device,
                          dOutFeatures, w_device, dw_device, rbB, nHotB,
                          input_nPlanes, input_stride,
                          output_nPlanes, output_stride);, w_device += c; dw_device += c;)
    cudaDeviceSynchronize();

    end = clock();
    time_gt = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
    dw_device -= c * _rules.size();
    w_device -= c * _rules.size();
    T *gt_dInFeatures_cpu = new T[inputFeatureSize * input_nPlanes];
    T *gt_dw_cpu = new T[input_nPlanes * output_nPlanes * 27];
    HANDLE_ERROR(cudaMemcpy( gt_dInFeatures_cpu, dInFeatures_device, inputFeatureSize * input_nPlanes * sizeof(T), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy( gt_dw_cpu, dw_device, input_nPlanes * output_nPlanes * 27 * sizeof(T), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    start = clock();
#endif

    dConvolution_backward_chunkbased((T *)inFeatures, (T *)dInFeatures,
                                        (T *)dOutFeatures , (T *)w, (T *)dw, (void *)inputAddress_device,new_validChunkCnt,
                                        input_nPlanes, input_stride,
                                        output_nPlanes, output_stride,_rules);

#if DEBUG_FAST_BACKWARD
    cudaDeviceSynchronize();
    end = clock();
    time_fast = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
    printf("computing time: %f %f\r\n", time_gt, time_fast);
    T *computed_dInFeatures_cpu = new T[inputFeatureSize * input_nPlanes];
    T *computed_dw_cpu = new T[input_nPlanes * output_nPlanes * 27];
    HANDLE_ERROR(cudaMemcpy( computed_dInFeatures_cpu, dInFeatures, inputFeatureSize * input_nPlanes * sizeof(T), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR(cudaMemcpy( computed_dw_cpu, dw, input_nPlanes * output_nPlanes * 27 * sizeof(T), cudaMemcpyDeviceToHost));
    for(int k = 0; k < inputFeatureSize * input_nPlanes; k++)
    {
        if(fabs(computed_dInFeatures_cpu[k] - gt_dInFeatures_cpu[k]) > 5e-2 * fabs(gt_dInFeatures_cpu[k])
                && fabs(computed_dInFeatures_cpu[k] - gt_dInFeatures_cpu[k]) > 1e-10)
        {
            printf("%d %f %f\r\n",k,computed_dInFeatures_cpu[k] * 1e10,gt_dInFeatures_cpu[k]* 1e10);
            assert(0);
        }
    }
    for(int k = 0; k < output_nPlanes * input_nPlanes * 27; k++)
    {
        if(fabs(computed_dw_cpu[k] - gt_dw_cpu[k]) > 5e-2 * fabs(gt_dw_cpu[k])
                && fabs(computed_dw_cpu[k] - gt_dw_cpu[k]) > 1e-7)
        {
            printf("criteria: %f %f %f\r\n",fabs(computed_dw_cpu[k] - gt_dw_cpu[k]) * 1e10, 5e-2 * fabs(gt_dw_cpu[k]* 1e10),
                   fabs(computed_dw_cpu[k] - gt_dw_cpu[k]) * 1e10);
            printf("%d %f %f\r\n",k,computed_dw_cpu[k] * 1e10,gt_dw_cpu[k]* 1e10);
            assert(0);
        }
    }
    printf("validation success!\r\n");
#endif
    // delete inputAddress;
    HANDLE_ERROR(cudaFree(inputAddress_device));

#if DEBUG_FAST_BACKWARD
    delete gt_dInFeatures_cpu,gt_dw_cpu,computed_dInFeatures_cpu,computed_dw_cpu;
    HANDLE_ERROR(cudaFree(dInFeatures_device));
    HANDLE_ERROR(cudaFree(dw_device));
    HANDLE_ERROR(cudaFree(w_device));
#endif
    }

}

#undef TACC
