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

#include "barney/common/half.h"
#include "barney/packedBSDF/PackedBSDF.h"

namespace BARNEY_NS {
  namespace render {
    
    struct Ray {
      inline __rtc_device void setVolumeHit(vec3f P, float t, vec3f albedo);
      inline __rtc_device PackedBSDF getBSDF() const;
      inline __rtc_device void setHit(vec3f P, vec3f N, float t,
                                    const PackedBSDF &packedBSDF);
      
      inline __rtc_device void makeShadowRay(vec3f _tp, vec3f _org, vec3f _dir, float len);
      inline __rtc_device bool hadHit() const { return bsdfType != PackedBSDF::NONE; }
      inline __rtc_device void clearHit(float newTMax = BARNEY_INF)
      { bsdfType = PackedBSDF::NONE; tMax = newTMax; }
      
      inline __rtc_device void packNormal(vec3f N);
      inline __rtc_device vec3f unpackNormal() const;
      inline __rtc_device vec3f getN() const  { return unpackNormal(); }
      
      vec3f    org;
      vec3f    dir;
      float    tMax;
      uint32_t rngSeed;

      union {
        struct {
          uint64_t misWeightBits:16;
          uint64_t pixelID      :27;
          /*! type of bsdf in the hitBSDF; if this is set to NONE the
            ray didn't have any hit yet */
          uint64_t bsdfType         : 3;
          uint64_t numDiffuseBounces: 2;
          /*! for path tracer: tracks whether we are, or aren't, in a
            refractable medium */
          uint64_t isInMedium : 1;
          uint64_t isSpecular : 1;
          uint64_t isShadowRay: 1;
          uint64_t dbg        : 1;
        };
        half     misWeight;
      };
      /*! the actual hit point, in 3D float coordinates (rather than
        implicitly through org+tMax*dir), for numerical robustness
        issues */
      vec3f    P;
      // vec3h       Le;
      vec3h    throughput;
      vec3h    N;
      union {
        PackedBSDF::Data hitBSDF;
        /*! the background color for primary rays that didn't have any intersection.
          do not use float4 here since that might incur padding for alignment reasosn.
         */
        bn_float4 missColor;
      };
    };
  
    inline __rtc_device PackedBSDF Ray::getBSDF() const
    {
      return PackedBSDF((PackedBSDF::Type)bsdfType,hitBSDF);
    }
    
    inline __rtc_device void Ray::setHit(vec3f P, vec3f N, float t,
                                       const PackedBSDF &packedBSDF)
    {
      this->packNormal(N);
      this->hitBSDF   = packedBSDF.data;
      this->bsdfType  = packedBSDF.type;
      this->tMax      = t;
      this->P         = P;
    }
    
    inline __rtc_device void Ray::setVolumeHit(vec3f P,
                                             float t,
                                             vec3f albedo)
    {
      setHit(P,vec3f(0.f),t,
             packedBSDF::Phase(albedo));
    }

    inline __rtc_device
    void Ray::makeShadowRay(vec3f _tp, vec3f _org, vec3f _dir, float len)
    {
      this->bsdfType = PackedBSDF::NONE;
      this->isShadowRay = true;
      this->dir = _dir;
      this->org = _org;
      this->throughput = _tp;
      this->tMax = len;
    }

    inline __rtc_device void Ray::packNormal(vec3f N)
    {
      this->N = N;
    }
  
    inline __rtc_device vec3f Ray::unpackNormal() const
    {
      return (vec3f)N;
    }
    
    inline __rtc_device
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
      return t0 <= t1;
    }

    inline __rtc_device
    bool boxTest(const Ray &ray,
                 range1f &tRange,
                 const box3f &box)
    {
      vec3f org = ray.org;
      vec3f dir = ray.dir;
      return boxTest(org,dir,tRange.lower,tRange.upper,box);
    }

    inline __rtc_device
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
      return t0 <= t1;
    }

    inline __rtc_device
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
}
