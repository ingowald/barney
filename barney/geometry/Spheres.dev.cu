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

#include "barney/geometry/Spheres.h"
#include "owl/owl_device.h"

namespace barney {
  
  OPTIX_BOUNDS_PROGRAM(SpheresBounds)(const void *geomData,                
                                     owl::common::box3f &bounds,  
                                     const int32_t primID)
  {
    const Spheres::DD &geom = *(const Spheres::DD *)geomData;
    vec3f origin = geom.origins[primID];
    bounds.lower = origin - geom.defaultRadius;
    bounds.upper = origin + geom.defaultRadius;
  }

  OPTIX_CLOSEST_HIT_PROGRAM(SpheresCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Spheres::DD>();
    int primID = optixGetPrimitiveIndex();

    ray.hadHit = true;
    ray.tMax = optixGetRayTmax();

    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    vec3f hitPos = org + ray.tMax * dir;
    vec3f center = self.origins[primID];
    float radius = self.defaultRadius;

    vec3f n = normalize(hitPos - center);
    vec3f baseColor = self.material.baseColor;//owl::randomColor(primID);
    ray.hit.baseColor = baseColor;//.3f + baseColor*abs(dot(dir,n));
    ray.hit.N = n;

    if (ray.dbg)
      printf("basecolor %f %f %f\n",
             baseColor.x,
             baseColor.y,
             baseColor.z);

#if 1
    vec3f P = org + ray.tMax * dir;
    ray.hit.P = center + (radius * 1.0001f) * normalize(P-center);
#else
    ray.hit.P = hitPos;
#endif
  }
  
  OPTIX_INTERSECT_PROGRAM(SpheresIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<Spheres::DD>();

    vec3f center = self.origins[primID];
    float radius = self.defaultRadius;
    
    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    const float tmin = optixGetRayTmin();
    float hit_t      = optixGetRayTmax();
    
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
    if (hit_t < optixGetRayTmax()) {
      optixReportIntersection(hit_t, 0);
    }
  }
  
}
