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

#include "barney/geometry/Attributes.dev.h"
#include "barney/geometry/Spheres.h"
#include "rtcore/TraceInterface.h"

// #include "owl/owl_device.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;
  
  inline __rtc_device
  float safe_eps(float f, vec3f v)
  {
    return max(f,1e-6f*reduce_max(abs(v)));
  }

  inline __rtc_device
  float safe_eps(float f, float v)
  {
    return max(f,1e-6f*fabsf(v));
  }
  
  
  struct SpheresPrograms {
    
    static inline __rtc_device
    void bounds(const rtc::TraceInterface &rt,
                const void *geomData,
                owl::common::box3f &bounds,  
                const int32_t primID)
    { 
      const Spheres::DD &geom = *(const Spheres::DD *)geomData;
      vec3f origin = geom.origins[primID];
      float radius = geom.radii?geom.radii[primID]:geom.defaultRadius;
      bounds.lower = origin - radius;
      bounds.upper = origin + radius;
    }
    
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &rt)
    {
      auto &ray = *(Ray*)rt.getPRD();
      auto &self = *(Spheres::DD*)rt.getProgramData();
      const World::DD &world = OptixGlobals::get(rt).world;
      int primID = rt.getPrimitiveIndex();
      int instID = rt.getInstanceIndex();

#ifdef NDEBUG
      bool dbg = false;
#else
      bool dbg = ray.dbg;
#endif
      
      float t_hit = rt.getRayTmax(); 

      vec3f org = rt.getWorldRayOrigin(); 
      vec3f dir = rt.getWorldRayDirection();
      /*! isec code has temporarily stored object-space hit position
        in ray.P, see below! */
      vec3f objectP = ray.P;
      vec3f worldP = rt.transformPointFromObjectToWorldSpace(objectP);
      
      vec3f objectCenter
        = self.origins[primID];
      vec3f objectN
        = (objectP == objectCenter)
        ? vec3f(1.f,0.f,0.f)
        : (objectP - objectCenter);
      float objectRadius = self.radii?self.radii[primID]:self.defaultRadius;

      /* shift object-space hit a bit away from the sphere */
      float eps = 1e-6f;
      eps = safe_eps(eps,objectRadius);
      eps = safe_eps(eps,objectP);
      objectP = objectCenter + (objectRadius+eps)*objectN;
    
      vec3f worldN
        = rt.transformVectorFromObjectToWorldSpace(objectN);

      render::HitAttributes hitData;
      hitData.worldPosition   = worldP;

      hitData.objectPosition  = objectP;
      hitData.worldNormal     = normalize(worldN);
      hitData.objectNormal    = normalize(objectN);
      hitData.primID          = primID;
      hitData.instID          = instID;
      hitData.t               = t_hit;
      if (self.colors)
        (vec3f&)hitData.color = self.colors[primID];
    
      auto interpolator = [&](const GeometryAttribute::DD &attrib) -> vec4f
      { // doesn't make sense, but anari sdk assumes for spheres
        // per-vtx is same as per-prim
        vec4f v = attrib.fromArray.valueAt(hitData.primID,dbg);
        return v;
      };
      self.setHitAttributes(hitData,interpolator,world,dbg);

      const DeviceMaterial &material
        = world.materials[self.materialID];
      material.setHit(ray,hitData,world.samplers,dbg);
    }
  
    static inline __rtc_device
    void anyHit(rtc::TraceInterface &rt)
    {
      /* nothing - already set in isec */
    }
  
    static inline __rtc_device
    void intersect(rtc::TraceInterface &rt)
    {
      const int primID = rt.getPrimitiveIndex();//optixGetPrimitiveIndex();
      const auto &self = *(const Spheres::DD*)rt.getProgramData();
      // = owl::getProgramData<Spheres::DD>();
      auto &ray = *(Ray*)rt.getPRD();//owl::getPRD<Ray>();
      
      vec3f center = self.origins[primID];
      float radius = self.radii?self.radii[primID]:self.defaultRadius;
      
      // with "move the origin" trick; see Ray Tracing Gems 2
      // const vec3f old_org  = optixGetObjectRayOrigin();
      const vec3f old_org  = rt.getObjectRayOrigin();
      // const vec3f dir  = optixGetObjectRayDirection();
      const vec3f dir  = rt.getObjectRayDirection();
      vec3f org = old_org;
      float t_move = max(0.f,length(center - old_org)-3.f*radius);
      org = org + t_move * dir;
      float t_max = rt.getRayTmax() - t_move;
      if (t_max < 0.f) return;
    
      float hit_t = t_max;

      float tmin = max(0.f,rt.getRayTmin()-t_move);
      const vec3f oc = org - center;
      const float a = dot(dir,dir);
      const float b = dot(oc, dir);
      const float c = dot(oc, oc) - radius * radius;
      const float discriminant = b * b - a * c;
    
      if (discriminant < 0.f) return;
    
      {
        float temp = (-b - sqrtf(discriminant)) / a;
        if (temp < hit_t && temp > tmin) 
          hit_t = temp;
      }
      
      {
        float temp = (-b + sqrtf(discriminant)) / a;
        if (temp < hit_t && temp > tmin) 
          hit_t = temp;
      }
      if (hit_t < t_max) {
        // "abuse" ray.P to store local sphere coordinate
        vec3f osPositionOfHit = /*shifted!*/org + /*shifted!*/hit_t*dir;
        ray.P = osPositionOfHit;
      
        hit_t += t_move;

        // vec3f P = old_org + hit_t * dir;
        // vec3f N = normalize(P-center);
        // P = optixTransformPointFromObjectToWorldSpace(center + radius * N);
        // N = optixTransformNormalFromObjectToWorldSpace(N);
        // vec3f geometryColor = NAN;
        // if (self.colors)
        //   geometryColor = self.colors[primID];
      
        // ray.setHit(P,N,hit_t,self.material,vec2f(NAN),geometryColor);

        // optixReportIntersection(hit_t, 0);
        rt.reportIntersection(hit_t, 0);
      }
    }
  };
  RTC_EXPORT_USER_GEOM(Spheres,Spheres::DD,SpheresPrograms,false,true);
}


