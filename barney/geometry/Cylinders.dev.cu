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

#include "barney/geometry/Cylinders.h"
#include "owl/owl_device.h"
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;

  /* ray - rounded cone intersection. */
  inline __device__
  bool intersectRoundedCone(const vec3f  pa, const vec3f  pb,
                            const float  ra, const float  rb,
                            const vec3f ray_org,
                            const vec3f ray_dir,
                            float& hit_t,
                            vec3f& isec_normal)
  {
    const vec3f& ro = ray_org;//ray.origin;
    const vec3f& rd = ray_dir;//ray.direction;

    vec3f  ba = pb - pa;
    vec3f  oa = ro - pa;
    vec3f  ob = ro - pb;
    float  rr = ra - rb;
    float  m0 = dot(ba, ba);
    float  m1 = dot(ba, oa);
    float  m2 = dot(ba, rd);
    float  m3 = dot(rd, oa);
    float  m5 = dot(oa, oa);
    float  m6 = dot(ob, rd);
    float  m7 = dot(ob, ob);

    float d2 = m0 - rr * rr;

    float k2 = d2 - m2 * m2;
    if (k2 == 0.f) return false;
    
    float k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
    float k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0 - m0 * ra * ra;

    float h = k1 * k1 - k0 * k2;
    if (h < 0.f) return false;
    float t = (-sqrtf(h) - k1) / k2;

    float y = m1 - ra * rr + t * m2;
    if (y > 0.f && y < d2)
      {
        hit_t = t;
        isec_normal = normalize(d2 * (oa + t * rd) - ba * y);
        return true;
      }

    // Caps. 
    float h1 = m3 * m3 - m5 + ra * ra;
    if (h1 > 0.0)
      {
        t = -m3 - sqrtf(h1);
        hit_t = t;
        isec_normal = normalize((oa + t * rd) / ra);
        return true;
      }
    return false;
  }
  
  // OPTIX_INTERSECT_PROGRAM(basicTubes_intersect)()
  // {
  //   const int primID = optixGetPrimitiveIndex();
  //   const auto& self
  //     = owl::getProgramData<TubesGeom>();

  //   owl::Ray ray(optixGetWorldRayOrigin(),
  //                optixGetWorldRayDirection(),
  //                optixGetRayTmin(),
  //                optixGetRayTmax());
  //   const Link link = self.links[primID];
  //   if (link.prev < 0) return;

  //   float tmp_hit_t = ray.tmax;

  //   vec3f pb, pa; float ra, rb;
  //   pa = link.pos;
  //   ra = link.rad;
  //   if (link.prev >= 0) {
  //     rb = self.links[link.prev].rad;
  //     pb = self.links[link.prev].pos;
  //     vec3f normal;

  //     if (intersectRoundedCone(pa, pb, ra,rb, ray, tmp_hit_t, normal))
  //       {
  //         if (optixReportIntersection(tmp_hit_t, primID)) {
  //           PerRayData& prd = owl::getPRD<PerRayData>();
  //           prd.linkID = primID;
  //           prd.t = tmp_hit_t;
  //           prd.isec_normal = normal;
  //         }
  //       }
  //   }
  // }

  struct CylindersPrograms {
  // OPTIX_BOUNDS_PROGRAM(CylindersBounds)(const void *geomData,
  //                                       owl::common::box3f &bounds,  
  //                                       const int32_t primID)

    static inline __rtc_device
    void bounds(const rtc::TraceInterface &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    {
      const Cylinders::DD &geom = *(const Cylinders::DD *)geomData;
      vec2i idx = geom.indices[primID];
      vec3f a = geom.vertices[idx.x];
      vec3f b = geom.vertices[idx.y];
      float ra, rb;
      ra = rb = geom.radii[primID];
      box3f box_a = {a-ra,a+ra};
      box3f box_b = {b-rb,b+rb};
      bounds.lower = min(box_a.lower,box_b.lower);
      bounds.upper = max(box_a.upper,box_b.upper);
    }
  
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
  
    static inline __rtc_device
    void intersect(rtc::TraceInterface &rt)
    // OPTIX_INTERSECT_PROGRAM(CylindersIsec)()
    {
      // capped
      const int primID
        = rt.getPrimitiveIndex();//optixGetPrimitiveIndex();
      const auto &self
        = *(Cylinders::DD*)rt.getProgramData();//owl::getProgramData<Cylinders::DD>();
      Ray &ray    = *(Ray*)rt.getPRD();
      
      const vec2i idx = self.indices[primID];
      const vec3f v0  = self.vertices[idx.x];
      const vec3f v1  = self.vertices[idx.y];
      
      const float radius
        = self.radii[primID];
      
      const vec3f ray_org  = rt.getObjectRayOrigin();//optixGetObjectRayOrigin();
      const vec3f ray_dir  = rt.getObjectRayDirection();//optixGetObjectRayDirection();
      float hit_t      = rt.getRayTmax();//optixGetRayTmax();
      const float ray_tmin = rt.getRayTmin();//optixGetRayTmin();
      const float ray_tmax = rt.getRayTmax();//optixGetRayTmax();
    
      const vec3f d = ray_dir;
      const vec3f s = v1 - v0; // axis
      const vec3f sxd = cross(s, d);
      const float a = dot(sxd, sxd); // (s x d)^2
      if (a == 0.f)
        return;
    
      const vec3f f = v0 - ray_org;
      const vec3f sxf = cross(s, f);
      const float ra = 1.0f/a;
      const float ts = dot(sxd, sxf) * ra; // (sd)(s x f) / (s x d)^2, in ray-space
      const vec3f fp = f - ts * d; // f' = v0 - closest point to axis
    
      const float s2 = dot(s, s); // s^2
      const vec3f perp = cross(s, fp); // s x f'
      const float c = radius*radius * s2 - dot(perp, perp); //  r^2 s^2 - (s x f')^2
      if (c < 0.f)
        return;

      float td = sqrtf(c * ra);
      const float tube_t0 = ts - td;
      const float tube_t1 = ts + td;
      
      // clip to cylinder caps
      const float sf = dot(s, f);
      const float sd = dot(s, d);

      float cap_t0 = -1e20f;
      float cap_t1 = -1e20f;
      if (sd == 0.f) {
        if (dot(ray_org-v0,v1-v0) < 0.f) return;
        if (dot(ray_org-v1,v0-v1) < 0.f) return;
      } else {
        const float rsd = 1.f/(sd);
        const float cap_t_v0 = sf * rsd;
        const float cap_t_v1 = cap_t_v0 + s2 * rsd;
        cap_t0 = min(cap_t_v0,cap_t_v1);
        cap_t1 = max(cap_t_v0,cap_t_v1);
      }
      
      // bool onCap_t0 = cap_t0 >= tube_t0;
      // bool onCap_t1 = cap_t1 <= tube_t1;
      const float t0 = max(cap_t0,tube_t0);
      const float t1 = min(cap_t1,tube_t1);
      if (t0 > t1) return;

      vec3f objectN = 0.f;
      if (ray_tmin <= t0 && t0 <= ray_tmax) {
        // front side hit:
        ray.tMax = t0;
        td *= -1.f;
        float hit_surf_u = (ray.tMax * sd - sf) * 1.f/(s2);
        objectN
          = (t0 == cap_t0)
          ? s
          : (td * d - fp - hit_surf_u * s);
      
      } else if (ray_tmin <= t1 && t1 <= ray_tmax) {
        ray.tMax = t1;
        float hit_surf_u = (ray.tMax * sd - sf) * 1.f/(s2);
        objectN
          = (t1 == cap_t1)
          ? -s
          : (td * d - fp - hit_surf_u * s);
      } else
        return;

      vec3f objectP = ray_org + ray.tMax * ray_dir;
      float t_hit = ray.tMax;

      float lerp_t
        = dot(objectP-v0,v1-v0)
        / (length(objectP-v0)*length(v1-v0));
      lerp_t = max(0.f,min(1.f,lerp_t));


      auto interpolator = [&](const GeometryAttribute::DD &attrib) -> float4
      {
        const vec4f value_a = attrib.fromArray.valueAt(idx.x);
        const vec4f value_b = attrib.fromArray.valueAt(idx.y);
        const vec4f ret = (1.f-lerp_t)*value_a + lerp_t*value_b;
        return ret;
      };

      render::HitAttributes hitData;
      hitData.worldPosition   = rt.transformPointFromObjectToWorldSpace(objectP);
      hitData.objectPosition  = objectP;
      hitData.worldNormal     = objectN;
      hitData.objectNormal    = rt.transformNormalFromObjectToWorldSpace(objectN);
      hitData.primID          = primID;
      hitData.t               = t_hit;;
    
      self.setHitAttributes(hitData,interpolator,ray.dbg);

      const DeviceMaterial &material
        = OptixGlobals::get(rt).materials[self.materialID];
      material.setHit(ray,hitData,OptixGlobals::get(rt).samplers,ray.dbg);
    
      rt.reportIntersection(ray.tMax, 0);
    }
  };
  
}

RTC_EXPORT_USER_GEOM(Cylinders,BARNEY_NS::CylindersPrograms,false,false);

