// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/common/rtcore-common.h"
#include <cuda_runtime.h>
#ifdef __CUDACC__
# include <cuda/std/limits>
# include <cuda.h>
#endif
#include "cuda-helper.h"

#define __rtc_device __device__
#define __rtc_both   __device__ __host__

#define RTC_HAVE_CUDA 1

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

