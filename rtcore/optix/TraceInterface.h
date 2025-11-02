// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <owl/owl.h>
#include "rtcore/cudaCommon/ComputeInterface.h"

namespace rtc {
  namespace optix {

    using namespace rtc::cuda_common;

// #ifdef __CUDACC__
#ifdef BARNEY_DEVICE_PROGRAM
    inline __device__
    const void *getLaunchParamsPointer();
#endif
    
    /*! the interface that pipeline programs use to talk to / query
      data from the ray tracing core */
    struct TraceInterface {
#ifdef BARNEY_DEVICE_PROGRAM
      inline __device__ const void *getLPData() const
      { return getLaunchParamsPointer(); }
#endif
#ifdef __CUDACC__
      
      inline __device__ void ignoreIntersection() const
      { optixIgnoreIntersection(); }
      
      inline __device__ void reportIntersection(float t, int i) const
      { optixReportIntersection(t,i); }

      inline __device__ void *getPRD() const
      { return ::owl::getPRDPointer(); }
      
      inline __device__ const void *getProgramData() const
      { return (const void *)optixGetSbtDataPointer(); }
      
      inline __device__ vec3i getLaunchDims()  const
      { return optixGetLaunchDimensions(); }
      
      inline __device__ vec3i getLaunchIndex() const
      { return optixGetLaunchIndex(); }

      inline __device__ vec2f getTriangleBarycentrics() const
      { return optixGetTriangleBarycentrics(); }
      
      inline __device__ int getPrimitiveIndex() const
      { return optixGetPrimitiveIndex(); }
      
      inline __device__ int getGeometryIndex() const
      { return optixGetSbtGASIndex(); }
      
      inline __device__ int getInstanceID() const
      { return optixGetInstanceId(); }
      
      inline __device__ int getRTCInstanceIndex() const
      { return optixGetInstanceIndex(); }
      
      inline __device__ float getRayTmax() const
      { return optixGetRayTmax(); }
      
      inline __device__ float getRayTmin() const
      { return optixGetRayTmin(); }

      inline __device__ vec3f getObjectRayDirection() const
      { return optixGetObjectRayDirection(); }
      
      inline __device__ vec3f getObjectRayOrigin() const
      { return optixGetObjectRayOrigin(); }
      
      inline __device__ vec3f getWorldRayDirection() const
      { return optixGetWorldRayDirection(); }
      
      inline __device__ vec3f getWorldRayOrigin() const
      { return optixGetWorldRayOrigin(); }
      
      inline __device__
      vec3f transformNormalFromObjectToWorldSpace(vec3f v) const
      { return optixTransformNormalFromObjectToWorldSpace(v); }
      
      inline __device__
      vec3f transformPointFromObjectToWorldSpace(vec3f v) const
      { return optixTransformPointFromObjectToWorldSpace(v); }
      
      inline __device__
      vec3f transformVectorFromObjectToWorldSpace(vec3f v) const
      { return optixTransformVectorFromObjectToWorldSpace(v); }
      
      inline __device__
      vec3f transformNormalFromWorldToObjectSpace(vec3f v) const
      { return optixTransformNormalFromWorldToObjectSpace(v); }
      
      inline __device__
      vec3f transformPointFromWorldToObjectSpace(vec3f v) const
      { return optixTransformPointFromWorldToObjectSpace(v); }
      
      inline __device__
      vec3f transformVectorFromWorldToObjectSpace(vec3f v) const
      { return optixTransformVectorFromWorldToObjectSpace(v); }
      
      
      inline __device__ void traceRay(rtc::AccelHandle world,
                                      vec3f org,
                                      vec3f dir,
                                      float t0,
                                      float t1,
                                      void *prdPtr) 
      {
        unsigned int           p0 = 0;
        unsigned int           p1 = 0;
        owl::packPointer(prdPtr,p0,p1);
        
        uint32_t rayFlags = 0u;
        owl::Ray ray(org,dir,t0,t1);
        optixTrace((const OptixTraversableHandle &)world,
                   (const ::float3&)ray.origin,
                   (const ::float3&)ray.direction,
                   ray.tmin,
                   ray.tmax,
                   ray.time,
                   ray.visibilityMask,
                   /*rayFlags     */ rayFlags,
                   /*SBToffset    */ ray.rayType,
                   /*SBTstride    */ ray.numRayTypes,
                   /*missSBTIndex */ ray.rayType,              
                   p0,
                   p1);
      }
#endif
    };

  }
}


#define RTC_DECLARE_GLOBALS(Type)                                       \
  extern "C" __constant__ Type optixLaunchParams;                       \
  inline __device__ const void *rtc::optix::getLaunchParamsPointer()    \
  { return &optixLaunchParams; }                                        \
    


#define RTC_EXPORT_USER_GEOM(name,DD,type,has_ah,has_ch)                \
                                                                        \
  extern "C" __global__                                                 \
  void __closesthit__##name() {                                         \
    ::rtc::optix::TraceInterface rtcore;                                \
    type::closestHit(rtcore);                                           \
  }                                                                     \
                                                                        \
  extern "C" __global__                                                 \
  void __anyhit__##name() {                                             \
    ::rtc::optix::TraceInterface rtcore;                                \
    type::anyHit(rtcore);                                               \
  }                                                                     \
                                                                        \
  extern "C" __global__                                                 \
  void __intersection__##name() {                                       \
    ::rtc::optix::TraceInterface rtcore;                                \
    type::intersect(rtcore);                                            \
  }                                                                     \
                                                                        \
  __device__ void __boundsFunc__##name(const void *geom,                \
                                       owl::common::box3f &result,      \
                                       int primID)                      \
  {                                                                     \
    ::rtc::optix::TraceInterface rtcore;                                \
    type::bounds(rtcore,geom,result,primID);                            \
  }                                                                     \
  extern "C" __global__                                                 \
  void __boundsFuncKernel__##name(const void *geom,                     \
                                  owl::common::box3f *boundsArray,      \
                                  int numPrims)                         \
  {                                                                     \
    uint32_t blockIndex                                                 \
      = blockIdx.x                                                      \
      + blockIdx.y * gridDim.x                                          \
      + blockIdx.z * gridDim.x * gridDim.y;                             \
    uint32_t primID                                                     \
      = threadIdx.x + blockDim.x*threadIdx.y                            \
      + blockDim.x*blockDim.y*blockIndex;                               \
    if (primID < numPrims) {                                            \
      __boundsFunc__##name(geom,boundsArray[primID],primID);            \
    }                                                                   \
  }                                                                     \
                                                                        \
    
  
#define RTC_EXPORT_TRIANGLES_GEOM(name,DD,type,has_ah,has_ch)   \
                                                                \
  extern "C" __global__                                         \
  void __closesthit__##name() {                                 \
    ::rtc::optix::TraceInterface rtcore;                        \
    type::closestHit(rtcore);                                   \
  }                                                             \
                                                                \
  extern "C" __global__                                         \
  void __anyhit__##name() {                                     \
    ::rtc::optix::TraceInterface rtcore;                        \
    type::anyHit(rtcore);                                       \
  }                                                             \
  
  

