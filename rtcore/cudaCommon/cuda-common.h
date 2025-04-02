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

#pragma once

#include "rtcore/common/rtcore-common.h"
#if BARNEY_HAVE_HIP
# include "hip/hip_runtime.h"
#define __CUDA_ARCH__ 1
# define cudaArray_t hipArray_t
# define cudaStream_t hipStream_t
# define cudaError_t hipError_t
# define cudaEvent_t hipEvent_t
# define cudaEventCreate hipEventCreate
# define cudaEventRecord hipEventRecord
# define cudaEventSynchronize hipEventSynchronize
# define cudaEventDestroy hipEventDestroy
# define cudaTextureFilterMode hipTextureFilterMode
# define cudaChannelFormatDesc hipChannelFormatDesc
# define cudaTextureAddressMode hipTextureAddressMode
# define cudaFilterModeLinear hipFilterModeLinear
# define cudaFilterModePoint hipFilterModePoint
# define cudaAddressModeMirror hipAddressModeMirror
# define cudaAddressModeClamp hipAddressModeClamp
# define cudaAddressModeWrap hipAddressModeWrap
# define cudaAddressModeBorder hipAddressModeBorder
# define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat
# define cudaReadModeElementType hipReadModeElementType
# define cudaCreateChannelDesc hipCreateChannelDesc
# define cudaChanneFormatlDesc hipChanneFormatlDesc
# define cudaExtent hipExtent
# define cudaMemset hipMemset
# define cudaMemsetAsync hipMemsetAsync
# define cudaAddressModifier hipAddressModifier
# define cudaResourceTypeArray hipResourceTypeArray
# define cudaTextureDesc hipTextureDesc
# define cudaResourceDesc hipResourceDesc
# define cudaMemcpyAsync hipMemcpyAsync
# define cudaMemcpy3DParms hipMemcpy3DParms
# define cudaMemcpyHostToDevice hipMemcpyHostToDevice
# define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
# define cudaSetDevice hipSetDevice
# define cudaGetDevice hipGetDevice
# define cudaGetDeviceCount hipGetDeviceCount
# define cudaMalloc hipMalloc
# define cudaMallocHost hipHostMalloc
# define cudaMallocArray hipMallocArray
# define cudaMalloc3DArray hipMalloc3DArray
# define cudaMallocManaged hipMallocManaged
# define cudaMallocAsync hipMallocAsync
# define make_cudaPitchedPtr make_hipPitchedPtr
# define cudaFreeAsync hipFreeAsync
# define cudaCreateTextureObject hipCreateTextureObject
# define cudaDestroyTextureObject hipDestroyTextureObject
# define cudaMemcpy hipMemcpy
# define cudaMemcpy3D hipMemcpy3D
# define cudaMemcpy2DToArray hipMemcpy2DToArray
# define cudaFree hipFree
# define cudaFreeArray hipFreeArray
# define cudaFreeHost hipFreeHost
# define cudaGetErrorString hipGetErrorString
# define cudaMemcpyDefault hipMemcpyDefault
# define cudaSuccess hipSuccess
# define cudaGetLastError hipGetLastError
# define cudaDeviceSynchronize hipDeviceSynchronize
# define cudaStreamCreate hipStreamCreate
# define cudaStreamSynchronize hipStreamSynchronize
# define cudaTextureObject_t hipTextureObject_t
# define cudaTextureReadMode hipTextureReadMode
# define CUDART_INF INFINITY
# define CUDART_INF_F ((float)INFINITY)
# define CUDART_NAN NAN
# define CUDART_NAN_F ((float)NAN)
#else
# include <cuda_runtime.h>
# ifdef __CUDACC__
#  include <cuda/std/limits>
#  include <cuda.h>
# endif
#endif
#include "cuda-helper.h"

#define __rtc_device __device__
#define __rtc_both   __device__ __host__

namespace rtc {
  namespace cuda_common {

    using namespace owl::common;    
    
    // ------------------------------------------------------------------
    // cuda vector types - import those into namesapce so we can
    // always disambiguate by writing rtc::float4 no matter what
    // backend we use
    // ------------------------------------------------------------------
    using ::float2;
    using ::float3;
    using ::float4;
    using ::int2;
    using ::int3;
    using ::int4;
    
    inline __both__ vec3f load(const float3 &v)
    { return vec3f(v.x,v.y,v.z); }
    inline __both__ vec4f load(const float4 &vv)
    { float4 v = vv; return vec4f(v.x,v.y,v.z,v.w); }
    
  }
}

