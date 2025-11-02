// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

// #include <cuda_runtime.h>
#ifdef __GNUC__
#   include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>

// Custom usleep function for Windows
void usleep(__int64 usec);

// Custom sleep function for Windows, emulating Unix sleep
void sleep(unsigned int seconds);

// Custom usleep function for Windows
inline void usleep(__int64 usec)
{
    // Convert microseconds to milliseconds (1 millisecond = 1000 microseconds)
    // Minimum sleep time is 1 millisecond
    __int64 msec = (usec / 1000 > 0) ? (usec / 1000) : 1;

    // Use the Sleep function from Windows API
    Sleep(static_cast<DWORD>(msec));
}

// Custom sleep function for Windows, emulating Unix sleep
inline void sleep(unsigned int seconds)
{
    // Convert seconds to milliseconds and call Sleep
    Sleep(seconds * 1000);
}
#endif

#ifdef NDEBUG
#define BARNEY_RAISE(MSG) throw std::runtime_error("fatal barney cuda error ... ")
#else
#define BARNEY_RAISE(MSG) { std::cerr << MSG << std::endl; assert(0); }
#endif


#define BARNEY_CUDA_CHECK( call )                                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
    printf("error code %i\n",rc); fflush(0);                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      BARNEY_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define BARNEY_CUDA_CALL(call) BARNEY_CUDA_CHECK(cuda##call)

#define BARNEY_CUDA_CHECK2( where, call )                                      \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      BARNEY_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define BARNEY_CUDA_SYNC_CHECK()                                       \
  {                                                             \
    BARNEY_CUDA_CALL(DeviceSynchronize());                      \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      BARNEY_RAISE("fatal cuda error");                            \
    }                                                           \
  }



#define BARNEY_CUDA_CHECK_NOTHROW( call )                                      \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#define BARNEY_CUDA_CALL_NOTHROW(call) BARNEY_CUDA_CHECK_NOTHROW(cuda##call)

#define BARNEY_CUDA_CHECK2_NOTHROW( where, call )                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#ifndef CHECK_CUDA_LAUNCH
# define CHECK_CUDA_LAUNCH(kernel,_nb,_bs,_shm,_s,...) \
  kernel<<<_nb,_bs,_shm,_s>>>(__VA_ARGS__);

//# define CHECK_CUDA_LAUNCH(kernel,_nb,_bs,_shm,_s,args...) \
//  kernel<<<_nb,_bs,_shm,_s>>>(args);
#endif
