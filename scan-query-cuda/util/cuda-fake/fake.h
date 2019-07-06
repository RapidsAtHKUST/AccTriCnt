//
// Created by yche on 1/31/18.
//

#ifndef CUDA_SCAN_FAKE_H
#define CUDA_SCAN_FAKE_H

#ifdef __JETBRAINS_IDE__

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

#include <climits>
#include "math.h"

// 1st: macros
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__

// 2nd: CUDA Synchronization
inline void __syncthreads() {};

inline void __threadfence_block() {};

inline void __threadfence() {};

inline void __threadfence_system();

inline int __syncthreads_count(int predicate) { return predicate; }

inline int __syncthreads_and(int predicate) { return predicate; }

inline int __syncthreads_or(int predicate) { return predicate; }

// 3rd: CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;

extern dim3 gridDim;
extern uint3 blockIdx;
extern dim3 blockDim;
extern uint3 threadIdx;
extern int warpsize;
extern int warpSize;

// 4th: Warp-Functions
// old:
DEPRECATED(int __any(int predicate));

// new:
int __all_sync(unsigned mask, int predicate);

int __any_sync(unsigned mask, int predicate);

unsigned __ballot_sync(unsigned mask, int predicate);

unsigned __activemask();

// 5th: Warp-Match Functions
template<class T>
unsigned int __match_any_sync(unsigned mask, T value);

template<class T>
unsigned int __match_all_sync(unsigned mask, T value, int *pred);

// Warp-Shuffle Functions
template<class T>
T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);

template<class T>
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);

template<class T>
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize);

template<class T>
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);

#include "fake_atomic.h"
#include "fake_math.h"

#endif

#endif //CUDA_SCAN_FAKE_H
