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
#include "barney/TiledFB.h"

namespace barney {

  struct Ray {
    vec3f    org;
    vec3f    dir;
    float    tMax;
    // these are only for debugging right now - eventually with ray
    // queue cycling there's no reaon why a ray should track those,
    // because they're not valid during shding, anyway, when the ray
    // gets shaded on another gpu than the one that found the
    // intersection:
    int      instID, geomID, primID;
    // only for debugging right now; should eventualy become
    // 'throughput' for a path tracer
    vec3f    color;
    float    u,v;
    uint32_t rngSeed;
    struct {
      uint32_t  pixelID:30;
      uint32_t  hadHit:1;
    };
    struct {
      uint32_t centerPixel:1;
      uint32_t dbg:1;
    };
  };

  struct RayQueue {
    struct DD {
      Ray *readQueue  = nullptr;
      
      /*! the queue where local kernels that write *new* rays
        (ie, ray gen and shading) will write their rays into */
      Ray *writeQueue = nullptr;
      
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
    Ray *readQueue  = nullptr;

    /*! the queue where local kernels that write *new* rays
      (ie, ray gen and shading) will write their rays into */
    Ray *writeQueue = nullptr;

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
      std::swap(readQueue, writeQueue);
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
      
      if (readQueue)  BARNEY_CUDA_CALL(Free(readQueue));
      if (writeQueue) BARNEY_CUDA_CALL(Free(writeQueue));

      if (!d_nextWritePos)
        BARNEY_CUDA_CALL(MallocManaged(&d_nextWritePos,sizeof(int)));
        
      BARNEY_CUDA_CALL(Malloc(&readQueue, newSize*sizeof(Ray)));
      BARNEY_CUDA_CALL(Malloc(&writeQueue,newSize*sizeof(Ray)));

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

}
