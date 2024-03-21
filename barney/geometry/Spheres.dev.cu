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
#include "owl/owl_device.h"

namespace barney {
  
  OPTIX_BOUNDS_PROGRAM(SpheresBounds)(const void *geomData,                
                                     owl::common::box3f &bounds,  
                                     const int32_t primID)
  {
    const Spheres::DD &geom = *(const Spheres::DD *)geomData;
    vec3f origin = geom.origins[primID];
    float radius = geom.radii?geom.radii[primID]:geom.defaultRadius;
    bounds.lower = origin - radius;
    bounds.upper = origin + radius;
  }

  OPTIX_CLOSEST_HIT_PROGRAM(SpheresCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Spheres::DD>();
    int primID = optixGetPrimitiveIndex();
    
    // ray.hadHit = true;
    float t_hit = optixGetRayTmax();

    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    vec3f P   = org + t_hit * dir;
    vec3f center = self.origins[primID];
    float radius = self.radii?self.radii[primID]:self.defaultRadius;
    vec3f N;
    if (P == center) {
      N = -normalize(dir);
    } else {
      N = normalize(P-center);
      P = center + (radius * 1.00001f) * N;
    }

    vec3f geometryColor(getColor(self,primID,primID,NAN,NAN/*no uv!*/));
    if (self.colors)
      geometryColor = self.colors[primID];
    ray.setHit(P,N,t_hit,self.material,vec2f(NAN),geometryColor);
    
    
    // ray.setHit(P,N,t_hit,mat);
  }
  
  OPTIX_INTERSECT_PROGRAM(SpheresIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<Spheres::DD>();

    vec3f center = self.origins[primID];
    float radius = self.radii?self.radii[primID]:self.defaultRadius;

#if 1
    // with "move the origin" trick; see Ray Tracing Gems 2
    const vec3f old_org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    vec3f org = old_org;
    float t_move = max(0.f,length(center - old_org)-10.f*radius);
    org = org + t_move * dir;
    float t_max = optixGetRayTmax() - t_move;
    if (t_max < 0.f) return;
    
    float hit_t = t_max;

    float tmin = max(0.f,optixGetRayTmin()-t_move);
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
    hit_t += t_move;
    if (hit_t < t_max) {
      optixReportIntersection(hit_t, 0);
    }
    
#else
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
#endif
  }
  
}
