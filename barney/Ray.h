// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/DeviceGroup.h"
/*! iw - TODO: this shouldn't be included here; we currently do
    because we ant to call 'ensureRayQueuesAReLargeEnoughFor(fb), but
    that should be handled somewhere else, not in the case ray class
    ... */
#include "barney/fb/TiledFB.h"
#include "barney/common/half.h"
#include "barney/common/Material.h"

namespace barney {

  struct Ray {
    vec3f    org;
    vec3h    throughput;
    vec3h    dir;
    float    tMax;
    uint32_t rngSeed;

    inline __device__ void setHit(vec3f P, vec3f N, float t,
                                  const Material::DD &material)
    {
      hit.P = P;
      hit.N = N;
      tMax = t;
      hadHit = true;
      hit.baseColor      = material.baseColor;
      hit.ior            = material.ior;
      hit.roughness      = material.roughness;
      hit.metallic       = material.metallic;
      hit.transmission   = material.transmission;
    }
    inline __device__ void setVolumeHit(vec3f P,
                                        float t,
                                        vec3f albedo)
    {
      hit.P = P;
      tMax = t;
      hadHit = true;
      hit.N = vec3f(0.f);
      hit.baseColor      = albedo;
      hit.ior            = 1.f;
      hit.transmission   = 0.f;
    }
    inline __device__ void makeShadowRay(vec3f tp, vec3f org, vec3f dir, float len)
    {
      this->hadHit = false;
      this->isShadowRay = true;
      this->dir = dir;
      this->org = org;
      this->throughput = tp;
      this->tMax = len;
    }
    
    struct {
      vec3h    N;
      vec3h    baseColor;
      vec3f    P;
      half     ior, transmission, metallic, roughness;
    } hit;
    struct {
      uint32_t  pixelID:28;
      uint32_t  hadHit:1;
      /*! for path tracer: tracks whether we are, or aren't, in a
          refractable medium */
      uint32_t  isInMedium:1;
      uint32_t  isShadowRay:1;
      uint32_t  dbg:1;
    };
  };

  struct RayQueue {
    struct DD {
      Ray *traceAndShadeReadQueue  = nullptr;
      
      /*! the queue where local kernels that write *new* rays
        (ie, ray gen and shading) will write their rays into */
      Ray *receiveAndShadeWriteQueue = nullptr;
      
      /*! current write position in the write queue (during shading and
        ray generation) */
      int *d_nextWritePos  = 0;
      int  numActive = 0;
      int  size     = 0;
    };
    
    RayQueue(Device *device) : device(device) {}

    /*! the read queue, where local kernels operating on rays (trace
      and shade) can read rays from. this is actually a misnomer
      becasue both shade and trace can actually modify trays (and
      thus, strictly speaking, are 'writing' to those rays), but
      haven't yet found a better name */
    Ray *traceAndShadeReadQueue  = nullptr;

    /*! the queue where local kernels that write *new* rays
      (ie, ray gen and shading) will write their rays into */
    Ray *receiveAndShadeWriteQueue = nullptr;

    /*! current write position in the write queue (during shading and
      ray generation) */
    int *d_nextWritePos  = 0;
    
    /*! how many rays are active in the *READ* queue */
    int numActiveRays() const { return numActive; }
    
    /*! how many rays are active in the *READ* queue */
    int  numActive = 0;
    int  size     = 0;

    Device *device = 0;

    void resetWriteQueue()
    {
      if (d_nextWritePos)
        *d_nextWritePos = 0;
    }
    
    void swap()
    {
      std::swap(receiveAndShadeWriteQueue, traceAndShadeReadQueue);
    }

    void ensureRayQueuesLargeEnoughFor(TiledFB *fb)
    {
      int requiredSize = fb->numActiveTiles * 2 * tileSize*tileSize;
      if (size >= requiredSize) return;
      resize(requiredSize);
    }
    
    void resize(int newSize)
    {
      assert(device);
      SetActiveGPU forDuration(device);
      
      if (traceAndShadeReadQueue)  BARNEY_CUDA_CALL(Free(traceAndShadeReadQueue));
      if (receiveAndShadeWriteQueue) BARNEY_CUDA_CALL(Free(receiveAndShadeWriteQueue));

      if (!d_nextWritePos)
        BARNEY_CUDA_CALL(MallocManaged(&d_nextWritePos,sizeof(int)));
        
      BARNEY_CUDA_CALL(Malloc(&traceAndShadeReadQueue, newSize*sizeof(Ray)));
      BARNEY_CUDA_CALL(Malloc(&receiveAndShadeWriteQueue,newSize*sizeof(Ray)));

      size = newSize;
    }
    
  };

  inline __device__
  bool boxTest(vec3f org, vec3f dir,
               float &t0, float &t1,
               const box3f &box)
  {
    vec3f t_lo = (box.lower - org) * rcp(dir);
    vec3f t_hi = (box.upper - org) * rcp(dir);
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);
    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    return t0 < t1;
  }

  inline __device__
  bool boxTest(const Ray &ray,
               range1f &tRange,
               const box3f &box)
  {
    vec3f org = ray.org;
    vec3f dir = ray.dir;
    return boxTest(org,dir,tRange.lower,tRange.upper,box);
  }

  inline __device__
  bool boxTest(float &t0, float &t1,
               box3f box,
               const vec3f org,
               const vec3f dir)
  {
    vec3f t_lo = (box.lower - org) * rcp(dir);
    vec3f t_hi = (box.upper - org) * rcp(dir);
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);
    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    return t0 < t1;
  }


  inline __device__
  bool boxTest(float &t0, float &t1,
               box4f box,
               const vec3f org,
               const vec3f dir)
  {
    return boxTest(t0,t1,box3f({box.lower.x,box.lower.y,box.lower.z},
                               {box.upper.x,box.upper.y,box.upper.z}),
                   org,dir);
  }

  
}
