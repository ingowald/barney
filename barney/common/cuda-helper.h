// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "owl/common.h"
#include <cuda_runtime.h>
#ifdef _GNUC_
#   include <unistd.h>
#endif
// inline void barneyRaise_impl(std::string str)
// {
//   fprintf(stderr,"%s\n",str.c_str());
// #ifdef WIN32
//   if (IsDebuggerPresent())
//     DebugBreak();
//   else
//     throw std::runtime_error(str);
// #else
// #ifndef NDEBUG
//   std::string bt = ::detail::backtrace();
//   fprintf(stderr,"%s\n",bt.c_str());
// #endif
//   raise(SIGINT);
// #endif
// }

#ifdef _WIN32
  #include <windows.h>

  // Custom usleep function for Windows
  void usleep(__int64 usec);

  // Custom sleep function for Windows, emulating Unix sleep
  void sleep(unsigned int seconds);
#else
  #include <unistd.h>  
#endif


#define BARNEY_RAISE(MSG) throw std::runtime_error("fatal barney cuda error ... ")
// #define BARNEY_RAISE(MSG) ::barneyRaise_impl(MSG);



#define BARNEY_CUDA_CHECK( call )                                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      printf("error code %i\n",rc); fflush(0);usleep(100);              \
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
    cudaDeviceSynchronize();                                    \
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


