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
#include <cuda_runtime.h>
#ifdef __CUDACC__
# include <cuda/std/limits>
# include <cuda.h>
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
    inline __both__ vec4f load(const float4 &v)
    { return vec4f(v.x,v.y,v.z,v.w); }
    
  }
}

