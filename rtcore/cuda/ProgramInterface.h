// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0



#pragma once

#include "rtcore/cuda/ComputeInterface.h"
#include "rtcore/cuda/Geom.h"

namespace rtc {
  namespace cuda {
    
    using rtc::cuda_common::ComputeInterface;

    struct Instance {
      affine3f objectToWorldXfm;
      affine3f worldToObjectXfm;
    };
    
    struct TraceInterface {
      inline __device__ void ignoreIntersection() 
      { rejectThisHit = true; }
      
      /*! the interface that pipeline programs use to talk to / query
        data from the ray tracing core */
      inline __device__ const void *getLaunchParamsPointer();
      
      inline __device__ void reportIntersection(float t, int i) 
      { current.tMax = t; }

      inline __device__ void *getPRD() const
      { return prd; }
      
      inline __device__ const void *getProgramData() const
      { return geomData; }
      
      inline __device__ const void *getLPData() const
      { return lpData; }
      
      inline __device__ vec3i getLaunchDims()  const
      {
        return vec3i(blockDim.x*gridDim.x,
                     blockDim.y*gridDim.y,
                     blockDim.z*gridDim.z);
      }
      
      inline __device__ vec3i getLaunchIndex() const
      {
        return vec3i(threadIdx.x+blockIdx.x*blockDim.x,
                     threadIdx.y+blockIdx.y*blockDim.y,
                     threadIdx.z+blockIdx.z*blockDim.z);
      }

      inline __device__ vec2f getTriangleBarycentrics() const
      { return current.triangleBarycentrics; }

      inline __device__ int getGeometryIndex() const
      { return current.geomID; }
      
      inline __device__ int getPrimitiveIndex() const
      { return current.primID; }
      
      inline __device__ int getInstanceID() const
      { return currentInstance->ID; }
      
      inline __device__ int getRTCInstanceIndex() const
      { return current.instID; }
      
      inline __device__ float getRayTmax() const
      { return current.tMax; }
      
      inline __device__ float getRayTmin() const
      { return tMin; }

      inline __device__ vec3f getObjectRayDirection() const
      { return object.dir; }
      
      inline __device__ vec3f getObjectRayOrigin() const
      { return object.org; }
      
      inline __device__ vec3f getWorldRayDirection() const
      { return world.dir; }
      
      inline __device__ vec3f getWorldRayOrigin() const
      { return world.org; }
      
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
      void traceRay(rtc::AccelHandle world,
                    vec3f org,
                    vec3f dir,
                    float t0,
                    float t1,
                    void *prdPtr);
      
      inline __device__
      bool intersectTriangle(const vec3f v0,const vec3f v1,const vec3f v2, bool dbg=false);
      
      // launch params
      const void  *lpData;
      
      // ray/traversal state:
      void  *prd;
      const void  *geomData;
      float  tMin;
      Geom::SBTHeader *acceptedSBT;
      struct {
        vec2f  triangleBarycentrics;
        int    primID;
        int    geomID;
        int    instID;
        float  tMax;
      } current, accepted;
      struct {
        vec3f org;
        vec3f dir;
      } world, object;
      const InstanceGroup::InstanceRecord *currentInstance;
      bool rejectThisHit;
    };

  }
}

#define RTC_DECLARE_GLOBALS(ignore) /* ignore */





#if RTC_DEVICE_CODE
# define RTC_CUDA_USERGEOM_KERNELS(name,type,has_ah,has_ch)             \
  __device__ void                                                       \
  _rtc_cuda_boundsFunc__##name(const void *geom,                        \
                               owl::common::box3f &result,              \
                               int primID)                              \
  {                                                                     \
    ::rtc::TraceInterface ti;                                           \
    type::bounds(ti,geom,result,primID);                                \
  }                                                                     \
  __global__ void                                                       \
  _rtc_cuda_boundsFuncKernel__##name(const void *geom,                  \
                                     owl::common::box3f *boundsArray,   \
                                     int numPrims)                      \
  {                                                                     \
    uint32_t blockIndex                                                 \
      = blockIdx.x                                                      \
      + blockIdx.y * gridDim.x                                          \
      + blockIdx.z * gridDim.x * gridDim.y;                             \
    uint32_t primID                                                     \
      = threadIdx.x + blockDim.x*threadIdx.y                            \
      + blockDim.x*blockDim.y*blockIndex;                               \
    if (primID < numPrims) {                                            \
      _rtc_cuda_boundsFunc__##name(geom,boundsArray[primID],primID);    \
    }                                                                   \
  }                                                                     \
  __global__                                                            \
  void rtc_cuda_writeAddresses_##name(rtc::Geom::SBTHeader *h)          \
  {                                                                     \
    h->ah = has_ah?type::anyHit:0;                                      \
    h->ch = has_ch?type::closestHit:0;                                  \
    h->user.intersect = type::intersect;                                \
  }                                                                     
#else
# define RTC_CUDA_USERGEOM_KERNELS(name,type,has_ah,has_ch)             \
  __global__ void                                                       \
  _rtc_cuda_boundsFuncKernel__##name(const void *geom,                  \
                                     owl::common::box3f *boundsArray,   \
                                     int numPrims);                     \
  __global__                                                            \
  void rtc_cuda_writeAddresses_##name(rtc::Geom::SBTHeader *h);
#endif
                                                



#define RTC_EXPORT_USER_GEOM(name,DD,type,has_ah,has_ch)                \
  RTC_CUDA_USERGEOM_KERNELS(name,type,has_ah,has_ch);                   \
  extern "C" void                                                       \
  _rtc_cuda_writeBounds__##name(rtc::Device *device,                    \
                                const void *geom,                       \
                                owl::common::box3f *boundsArray,        \
                                int numPrims)                           \
  {                                                                     \
    if (numPrims == 0) return;                                          \
    int bs = 1024;                                                      \
    int nb = owl::common::divRoundUp(numPrims,bs);                      \
    _rtc_cuda_boundsFuncKernel__##name<<<nb,bs,0,device->stream>>>      \
      (geom,boundsArray,numPrims);                                      \
  }                                                                     \
  rtc::GeomType *createGeomType_##name(rtc::Device *device)             \
  {                                                                     \
    ::rtc::SetActiveGPU forDuration(device);                            \
    rtc::Geom::SBTHeader *h;                                            \
    BARNEY_CUDA_CALL(Malloc((void **)&h,sizeof(*h)));                   \
    rtc_cuda_writeAddresses_##name<<<1,32>>>(h);                        \
    device->sync();                                                     \
    rtc::Geom::SBTHeader hh;                                            \
    BARNEY_CUDA_CALL(Memcpy(&hh,h,sizeof(hh),cudaMemcpyDefault));       \
    BARNEY_CUDA_CALL(Free(h));                                          \
    return new rtc::cuda::UserGeomType                                  \
      (device,                                                          \
       sizeof(DD),                                                      \
       _rtc_cuda_writeBounds__##name,                                   \
       hh.user.intersect,                                               \
       hh.ah,                                                           \
       hh.ch                                                            \
       );                                                               \
  }


#if RTC_DEVICE_CODE
# define RTC_CUDA_TRIANGLES_WRITEADDR(name,Programs,has_ah,has_ch)      \
  __global__                                                            \
  void rtc_cuda_writeAddresses_##name(rtc::Geom::SBTHeader *h)          \
  {                                                                     \
    h->ah = has_ah?Programs::anyHit:0;                                  \
    h->ch = has_ch?Programs::closestHit:0;                              \
  }                                                                     
#else
# define RTC_CUDA_TRIANGLES_WRITEADDR(name,Programs,has_ah,has_ch)      \
  __global__                                                            \
  void rtc_cuda_writeAddresses_##name(rtc::Geom::SBTHeader *h);
#endif

#define RTC_EXPORT_TRIANGLES_GEOM(name,DD,Programs,has_ah,has_ch)       \
  RTC_CUDA_TRIANGLES_WRITEADDR(name,Programs,has_ah,has_ch)             \
  rtc::GeomType *createGeomType_##name(rtc::Device *device)             \
  {                                                                     \
    ::rtc::SetActiveGPU forDuration(device);                            \
    rtc::Geom::SBTHeader *h;                                            \
    BARNEY_CUDA_CALL(Malloc((void **)&h,sizeof(*h)));                   \
    rtc_cuda_writeAddresses_##name<<<1,32>>>(h);                        \
    device->sync();                                                     \
    rtc::Geom::SBTHeader hh;                                            \
    BARNEY_CUDA_CALL(Memcpy(&hh,h,sizeof(hh),cudaMemcpyDefault));       \
    return new rtc::TrianglesGeomType                                   \
      (device,                                                          \
       sizeof(DD),                                                      \
       hh.ah,                                                           \
       hh.ch);                                                          \
  }
