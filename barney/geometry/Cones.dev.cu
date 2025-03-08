// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/geometry/Cones.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;

  inline __rtc_device float sqrt(float f) { return sqrtf(f); }
  inline __rtc_device float inversesqrt(float f) { return 1./sqrtf(f); }
  inline __rtc_device float length2(vec3f v) { return dot(v,v); }
  
  struct ConesPrograms {
    /*! bounding box program */
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      auto &self = *(Cones::DD*)geomData;
      const vec2i pidx
        = self.indices
        ? self.indices[primID]
        : (2 * primID + vec2i(0, 1));
    
      const vec3f pa = (const vec3f&)self.vertices[pidx.x];
      const vec3f pb = (const vec3f&)self.vertices[pidx.y];
    
      const float ra = self.radii[pidx.x];
      const float rb = self.radii[pidx.y];
      box3f aBox(pa-ra,pa+ra);
      box3f bBox(pb-ra,pb+rb);
    
      bounds.lower = min(aBox.lower,bBox.lower);
      bounds.upper = max(aBox.upper,bBox.upper);
    }

    /*! closest-hit program - doesn't do anything because we do all the
      work in IS prog, but needs to exist to make optix happy */
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &rt)
    {
      /* nothing - already set in isec */
    }
    
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &rt)
    {
      /* nothing - already set in isec */
    }
  
    /*! largely stolen from VisRTX */
    static inline __rtc_device
    void intersect(rtc::TraceInterface &rt)
    {
      Ray &ray    = *(Ray*)rt.getPRD();
      const auto &self
        = *(Cones::DD*)rt.getProgramData();
      int primID = rt.getPrimitiveIndex();
        
      render::HitAttributes hitData;
      const DeviceMaterial &material
        = OptixGlobals::get(rt).materials[self.materialID];
      hitData.t = ray.tMax;
      
      vec3f ro  = rt.getObjectRayOrigin();
      vec3f rd  = rt.getObjectRayDirection();
    
      const vec2i idx
        = self.indices
        ? self.indices[primID]
        : (2 * primID + vec2i(0, 1));
    
      const auto p0 = (const vec3f &)self.vertices[idx.x];
      const auto p1 = (const vec3f &)self.vertices[idx.y];

      const float ra = self.radii[idx.x];
      const float rb = self.radii[idx.y];

      const vec3f ba = p1 - p0;
      const vec3f oa = ro - p0;
      const vec3f ob = ro - p1;

      const float m0 = dot(ba, ba);
      const float m1 = dot(oa, ba);
      const float m2 = dot(ob, ba);
      const float m3 = dot(rd, ba);

      float lerp_t = 0.f;

      // interpolator for anari-style color/attribute interpolation
      auto interpolator = [&](const GeometryAttribute::DD &attrib) -> vec4f
      {
        const vec4f value_a = attrib.fromArray.valueAt(idx.x);
        const vec4f value_b = attrib.fromArray.valueAt(idx.y);
        const vec4f ret = (1.f-lerp_t)*value_a + lerp_t*value_b;
        return ret;
      };

      
      if (m1 < 0.0f) {
        if (length2(oa * m3 - rd * m1) < (ra * ra * m3 * m3)) {
          lerp_t = 0.f;
          float t = -m1 / m3;
          if (t < hitData.t) {
          vec3f N = normalize(-ba * inversesqrt(m0));
          vec3f P = (vec3f)ro+t*rd;

          hitData.primID          = primID;
          hitData.t               = t;
          hitData.objectPosition  = P;
          hitData.objectNormal    = normalize(N);
          hitData.worldPosition
            = rt.transformPointFromObjectToWorldSpace(P);
          hitData.worldNormal
            = normalize(rt.transformNormalFromObjectToWorldSpace(N));
          
          // trigger the anari attribute evaluation
          self.setHitAttributes(hitData,interpolator,ray.dbg);
          
          // ... store the hit in the ray, rqs-style ...
          material.setHit(ray,hitData,OptixGlobals::get(rt).samplers,ray.dbg);
          
          // .... and let optix know we did have a hit.
          rt.reportIntersection(hitData.t, 0);
          }
        }
      } else if (m2 > 0.0f) {
        lerp_t = 1.f;
        if (length2(ob * m3 - rd * m2) < (rb * rb * m3 * m3)) {
          float t = -m2 / m3;
          if (t < hitData.t) {
          vec3f N = normalize(ba * inversesqrt(m0));
          vec3f P = (vec3f)ro+t*rd;

          hitData.primID          = primID;
          hitData.t               = t;
          hitData.objectPosition  = P;
          hitData.objectNormal    = normalize(N);
          hitData.worldPosition
            = rt.transformPointFromObjectToWorldSpace(P);
          hitData.worldNormal
            = normalize(rt.transformNormalFromObjectToWorldSpace(N));
                    
          // trigger the anari attribute evaluation
          self.setHitAttributes(hitData,interpolator,ray.dbg);
          
          // ... store the hit in the ray, rqs-style ...
          material.setHit(ray,hitData,OptixGlobals::get(rt).samplers,ray.dbg);
          
          // .... and let optix know we did have a hit.
          rt.reportIntersection(hitData.t, 0);
          }
        }
      }
      
      const float m4 = dot(rd, oa);
      const float m5 = dot(oa, oa);
      const float rr = ra - rb;
      const float hy = m0 + rr * rr;

      float k2 = m0 * m0 - m3 * m3 * hy;
      float k1 = m0 * m0 * m4 - m1 * m3 * hy + m0 * ra * (rr * m3 * 1.0f);
      float k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0f - m0 * ra);

      const float h = k1 * k1 - k2 * k0;
      if (h < 0.0f)
        return;

      const float t = (-k1 - sqrt(h)) / k2;
      const float y = m1 + t * m3;
      if (y > 0.0f && y < m0 && t < hitData.t) {
        vec3f N = normalize(m0 * (m0 * (oa + t * rd) + rr * ba * ra) - ba * hy * y);
        vec3f P = (vec3f)ro+t*rd;
        lerp_t = y / m0;

        hitData.primID          = primID;
        hitData.t               = t;
        hitData.objectPosition  = P;
        hitData.objectNormal    = normalize(N);
        hitData.worldPosition
          = rt.transformPointFromObjectToWorldSpace(P);
        hitData.worldNormal
          = normalize(rt.transformNormalFromObjectToWorldSpace(N));
        
        // trigger the anari attribute evaluation
        self.setHitAttributes(hitData,interpolator,ray.dbg);
        
        // ... store the hit in the ray, rqs-style ...
        material.setHit(ray,hitData,OptixGlobals::get(rt).samplers,ray.dbg);
        
        // .... and let optix know we did have a hit.
        rt.reportIntersection(hitData.t, 0);
      }
    }
  };
  
  RTC_EXPORT_USER_GEOM(Cones,Cones::DD,ConesPrograms,false,false);
} // ::BARNEY_NS

