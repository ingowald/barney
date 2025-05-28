// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

/*! file rtcore/cudaCommon/ComputeInterface.h Abstracts device-side
    compute operations (like atomics, texture-sampling, etc), defines
    the interace with which compute kernels can talk to the device (to
    get thread idx etc), and the EXPORT macros to define device
    kernels that host code can later import (using the
    ComputeKernel.h:IMPORT macros) */
#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace cuda_common {
    
    // ==================================================================
    // INTERFACE
    // ==================================================================

    inline __device__ void fatomicMin(float *addr, float value);
    inline __device__ void fatomicMax(float *addr, float value);

// #if RTC_DEVICE_CODE
    /* texturing wrappers; can only be instantiated for 'flaot' and
       'vec4f' types */
    template<typename T> inline __device__
    T tex1D(rtc::TextureObject to, float x);
    
    /* texturing wrappers; can only be instantiated for 'flaot' and
       'vec4f' types */
    template<typename T> inline __device__
    T tex2D(rtc::TextureObject to, float x, float y);

    /* texturing wrappers; can only be instantiated for 'flaot' and
       'vec4f' types */
    template<typename T> inline __device__
    T tex3D(rtc::TextureObject to, float x, float y, float z);
// #endif
    
    struct ComputeInterface
    {
// #if RTC_DEVICE_CODE
#ifdef __CUDACC__
      inline __device__ vec3ui launchIndex() const
      {
        return getThreadIdx() + getBlockIdx() * getBlockDim();
      }
      inline __device__ vec3ui getThreadIdx() const
      { return vec3ui(threadIdx.x,threadIdx.y,threadIdx.z); }
      inline __device__ vec3ui getBlockDim() const
      { return {blockDim.x,blockDim.y,blockDim.z}; }
      inline __device__ vec3ui getBlockIdx() const
      { return {blockIdx.x,blockIdx.y,blockIdx.z}; }
      inline __device__ int atomicAdd(int *ptr, int inc) const
      { return ::atomicAdd(ptr,inc); }
      inline __device__ float atomicAdd(float *ptr, float inc) const
      { return ::atomicAdd(ptr,inc); }
#endif
    };
    
    // ==================================================================
    // INLINE IMPLEMENTATION
    // ==================================================================
// #if RTC_DEVICE_CODE
#ifdef __CUDACC__
    // ------------------------------------------------------------------
    // cuda texturing
    // ------------------------------------------------------------------

    template<> inline __device__
    float tex2D<float>(rtc::TextureObject to, float x, float y)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      return ::tex2D<float>(texObj,x,y);
    }

    template<> inline __device__
    float tex3D<float>(rtc::TextureObject to, float x, float y, float z)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      float f= ::tex3D<float>(texObj,x,y,z);
      return f;
    }

    template<> inline __device__
    vec4f tex1D<vec4f>(rtc::TextureObject to, float x)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex1D<::float4>(texObj,x);
      return load(v);
    }
    
    template<> inline __device__
    vec4f tex2D<vec4f>(rtc::TextureObject to, float x, float y)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex2D<::float4>(texObj,x,y);
      return load(v);
    }

    template<> inline __device__
    vec4f tex3D<vec4f>(rtc::TextureObject to, float x, float y, float z)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex3D<::float4>(texObj,x,y,z);
      return load(v);
    }
    
    // ------------------------------------------------------------------
    // atomics
    // ------------------------------------------------------------------
    using ::atomicCAS;

    inline __device__
    void fatomicMin(float *addr, float value)
    {
      float old = *(volatile float *)addr;
      if(old <= value) return;

      int _expected = __float_as_int(old);
      int _desired  = __float_as_int(value);
      while (true) {
        uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
        if (_found == _expected)
          // write went though; we _did_ write the new mininm and
          // are done.
          return;
        // '_expected' changed, so write did not go through, and
        // somebody else wrote something new to that location.
        old = __int_as_float(_found);
        if (old <= value)
          // somebody else wrote something that's already smaller
          // than what we have ... leave it be, and done.
          return;
        else {
          // somebody else wrote something, but ours is _still_ smaller.
          _expected = _found;
          continue;
        }
      } 
    }

    inline __device__
    void fatomicMax(float *addr, float value)
    {
      float old = *(volatile float *)addr;
      if(old >= value) return;

      int _expected = __float_as_int(old);
      int _desired  = __float_as_int(value);
      while (true) {
        uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
        if (_found == _expected)
          // write went though; we _did_ write the new mininm and
          // are done.
          return;
        // '_expected' changed, so write did not go through, and
        // somebody else wrote something new to that location.
        old = __int_as_float(_found);
        if (old >= value)
          // somebody else wrote something that's already smaller
          // than what we have ... leave it be, and done.
          return;
        else {
          // somebody else wrote something, but ours is _still_ smaller.
          _expected = _found;
          continue;
        }
      } 
    }
#endif
    
  }
}



#if RTC_DEVICE_CODE
# define RTC_CUDA_KERNEL(name,className)                                \
  __global__ void _barney_cuda_kernel_##name(className dd)              \
  {                                                                     \
    rtc::cuda_common::ComputeInterface ci;                              \
    dd.run(ci);                                                         \
  }                                                                     
#else
# define RTC_CUDA_KERNEL(name,className)                                \
  __global__ void _barney_cuda_kernel_##name(className dd);
#endif

#define RTC_EXPORT_COMPUTE1D(name,className)                            \
  RTC_CUDA_KERNEL(name,className)                                       \
  struct _barney_cuda_compute1D_##name                                  \
    : public rtc::cuda_common::ComputeKernel1D                          \
  {                                                                     \
    _barney_cuda_compute1D_##name(rtc::Device *device)                  \
      : device(device)                                                  \
      {}                                                                \
    void launch(unsigned int nb,                                        \
                unsigned int bs,                                        \
                const void *ddPtr) override                             \
    {                                                                   \
      ::rtc::cuda_common::SetActiveGPU forDuration(device);             \
      if (nb > 0)                                                       \
      _barney_cuda_kernel_##name<<<nb,bs,0,device->stream>>>            \
        (*(const className*)ddPtr);                                     \
    }                                                                   \
    rtc::Device *const device;                                          \
  };                                                                    \
  rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev)          \
  { return new _barney_cuda_compute1D_##name(dev); }                    \
  
#define RTC_EXPORT_COMPUTE2D(name,className)                            \
  RTC_CUDA_KERNEL(name,className)                                       \
  struct _barney_cuda_compute2D_##name                                  \
    : public rtc::cuda_common::ComputeKernel2D                          \
  {                                                                     \
    _barney_cuda_compute2D_##name(rtc::Device *device)                  \
      : device(device)                                                  \
      {}                                                                \
    void launch(::rtc::vec2ui nb,                                       \
                ::rtc::vec2ui bs,                                       \
                const void *ddPtr) override                             \
    {                                                                   \
      ::rtc::cuda_common::SetActiveGPU forDuration(device);             \
      if (nb.x > 0 && nb.y > 0)                                         \
      _barney_cuda_kernel_##name                                        \
        <<<dim3{(unsigned)nb.x,(unsigned)nb.y,1u},                      \
        dim3{(unsigned)bs.x,(unsigned)bs.y,1u},                         \
        0,device->stream>>>                                             \
        (*(const className*)ddPtr);                                     \
    }                                                                   \
    rtc::Device *const device;                                          \
  };                                                                    \
  rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev)          \
  { return new _barney_cuda_compute2D_##name(dev); }                    \

#define RTC_EXPORT_COMPUTE3D(name,className)                            \
  RTC_CUDA_KERNEL(name,className)                                       \
  struct _barney_cuda_compute3D_##name                                  \
    : public rtc::cuda_common::ComputeKernel3D                          \
  {                                                                     \
    _barney_cuda_compute3D_##name(rtc::Device *device)                  \
      : device(device)                                                  \
      {}                                                                \
    void launch(rtc::vec3ui nb,                                         \
                rtc::vec3ui bs,                                         \
                const void *ddPtr) override                             \
    {                                                                   \
      ::rtc::cuda_common::SetActiveGPU forDuration(device);             \
      if (nb.x > 0 && nb.y > 0 && nb.z > 0)                             \
      _barney_cuda_kernel_##name                                        \
        <<<dim3{(unsigned)nb.x,(unsigned)nb.y,(unsigned)nb.z},          \
        dim3{(unsigned)bs.x,(unsigned)bs.y,(unsigned)bs.z},             \
        0,device->stream>>>                                             \
        (*(const className*)ddPtr);                                     \
    }                                                                   \
    rtc::Device *const device;                                          \
  };                                                                    \
  rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev)          \
  { return new _barney_cuda_compute3D_##name(dev); }                    \


