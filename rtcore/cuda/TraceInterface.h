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

#include <owl/owl.h>
#include "rtcore/cudaCommon/ComputeInterface.h"
#include "rtcore/cuda/TraceKernel.h"

#if !BARNEY_DEVICE_PROGRAM
# error "RTcore.h should only ever be included by device programs"
#endif

namespace rtc {
  namespace cuda {

    using namespace rtc::cuda_common;

    struct Instance {
      affine3f objectToWorldXfm;
      affine3f worldToObjectXfm;
    };
    
    /*! the interface that pipeline programs use to talk to / query
      data from the ray tracing core */
    inline __device__ const void *getLaunchParamsPointer();
    struct TraceInterface {
      inline __device__ void ignoreIntersection() 
      { ignoreThisHit = true; }
      
      inline __device__ void reportIntersection(float t, int i) 
      { hit_t = t; }

      inline __device__ void *getPRD() const
      { return prd; }
      
      inline __device__ const void *getProgramData() const
      { return geomData; }
      
      inline __device__ const void *getLPData() const
      { return lpData; }
      
      inline __device__ vec3i getLaunchDims()  const
      {
        return (const vec3i&)blockDim * (const vec3i &)blockIdx;
      }
      
      inline __device__ vec3i getLaunchIndex() const
      { return (const vec3i&)threadIdx + (const vec3i&)blockIdx*(const vec3i&)blockDim; }

      inline __device__ vec2f getTriangleBarycentrics() const
      { return triangleBarycentrics; }
      
      inline __device__ int getPrimitiveIndex() const
      { return primID; }
      
      inline __device__ float getRayTmax() const
      { return tMax; }
      
      inline __device__ float getRayTmin() const
      { return tMin; }

      inline __device__ vec3f getObjectRayDirection() const
      { return object.direction; }
      
      inline __device__ vec3f getObjectRayOrigin() const
      { return object.origin; }
      
      inline __device__ vec3f getWorldRayDirection() const
      { return world.direction; }
      
      inline __device__ vec3f getWorldRayOrigin() const
      { return world.origin; }
      
      inline __device__
      vec3f transformNormalFromObjectToWorldSpace(vec3f v) const;
      
      inline __device__
      vec3f transformPointFromObjectToWorldSpace(vec3f v) const;
      
      inline __device__
      vec3f transformVectorFromObjectToWorldSpace(vec3f v) const;
      
      inline __device__
      vec3f transformNormalFromWorldToObjectSpace(vec3f v) const;
      
      inline __device__
      vec3f transformPointFromWorldToObjectSpace(vec3f v) const;
      
      inline __device__
      vec3f transformVectorFromWorldToObjectSpace(vec3f v) const;
      
      inline __device__
      void traceRay(rtc::device::AccelHandle world,
                    vec3f org,
                    vec3f dir,
                    float t0,
                    float t1,
                    void *prdPtr);
      
      float           hit_t;
      void           *prd;
      void           *geomData;
      void           *lpData;
      vec2f           triangleBarycentrics;
      float tMin;
      float tMax;
      int   primID;
      struct {
        vec3f origin;
        vec3f direction;
      } world, object;
      const Instance *currentInstance;
      bool ignoreThisHit;
    };

    
    inline __device__
    void TraceInterface::traceRay(rtc::device::AccelHandle world,
                                  vec3f org,
                                  vec3f dir,
                                  float t0,
                                  float t1,
                                  void *prdPtr)
    {
      printf("trace ...\n");
    }
    
    inline __device__
    vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
    {
      return xfmVector(currentInstance->objectToWorldXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmPoint(currentInstance->objectToWorldXfm,
                      (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmVector(currentInstance->objectToWorldXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
    {
      return xfmVector(currentInstance->worldToObjectXfm.l,
                       (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmPoint(currentInstance->worldToObjectXfm,
                      (const owl::common::vec3f &)v);
    }

    inline __device__
    vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmVector(currentInstance->worldToObjectXfm.l,
                       (const owl::common::vec3f &)v);
    }
    
  }
}

#define RTC_DECLARE_GLOBALS(ignore) /* ignore */

#define RTC_EXPORT_TRACE2D(name,Class)                          \
  __global__                                                    \
  void rtc_cuda_run_##name(::rtc::cuda::TraceInterface ti)      \
  {                                                             \
    Class::run(ti);                                             \
  }                                                             \
  void rtc_cuda_launch_##name(rtc::Device *device,              \
                              vec2i dims,                       \
                              const void *dd)                   \
  {                                                             \
    ::rtc::cuda_common::SetActiveGPU forDuration(device);       \
    vec2i bs(8,8);                                              \
    vec2i nb = divRoundUp(dims,bs);                             \
    ::rtc::cuda::TraceInterface ti;                             \
    rtc_cuda_run_##name<<<nb,bs>>>(ti);                         \
  }                                                             \
  ::rtc::TraceKernel2D *createTrace_##name(rtc::Device *device) \
  {                                                             \
    return new ::rtc::cuda::TraceKernel2D                       \
      (device,sizeof(Class),rtc_cuda_launch_##name);            \
  }                                                             \
  
