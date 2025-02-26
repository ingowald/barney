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

// #include <cuda_runtime.h>
#ifdef __GNUC__
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


#define BARNEY_RAISE(MSG) throw std::runtime_error("fatal barney cuda error ... ")
// #define BARNEY_RAISE(MSG) ::barneyRaise_impl(MSG);



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

#ifndef CHECK_CUDA_LAUNCH
# define CHECK_CUDA_LAUNCH(kernel,_nb,_bs,_shm,_s,...) \
  kernel<<<_nb,_bs,_shm,_s>>>(__VA_ARGS__);

//# define CHECK_CUDA_LAUNCH(kernel,_nb,_bs,_shm,_s,args...) \
//  kernel<<<_nb,_bs,_shm,_s>>>(args);
#endif
