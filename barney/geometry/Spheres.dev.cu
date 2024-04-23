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
#include "barney/material/Inputs.h"
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

  inline __device__
  float safe_eps(float f, vec3f v)
  {
    return max(f,1e-6f*reduce_max(abs(v)));
  }

  inline __device__
  float safe_eps(float f, float v)
  {
    return max(f,1e-6f*fabsf(v));
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(SpheresCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Spheres::DD>();
    int primID = optixGetPrimitiveIndex();
    
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
      float eps = 1e-6f;
      eps = safe_eps(eps,radius);
      eps = safe_eps(eps,P);
      
      float offset = radius*(1.f+eps);
      P = center + offset * N;
    }

    // THIS IS WRONG: !!!!!!!!!
    if (ray.dbg) printf("storing wrong normals here!\n");
    
    render::HitAttributes hitData(OptixGlobals::get());
    hitData.worldPosition   = P;
    hitData.objectPosition  = P;
    hitData.worldNormal     = N;
    hitData.objectNormal    = N;
    hitData.primID          = primID;
    hitData.t               = t_hit;
    if (self.colors)
      (vec3f&)hitData.color = self.colors[primID];

    auto interpolate = [&](const render::GeometryAttribute::DD &)
    { /* does not make sense for spheres */return make_float4(0,0,0,1); };
    self.evalAttributesAndStoreHit(ray,hitData,interpolate);
    
    // vec3f geometryColor(getColor(self,primID,primID,NAN,NAN/*no uv!*/));
    // if (self.colors)
    //   geometryColor = self.colors[primID];
    // ray.setHit(P,N,t_hit,self.material,vec2f(NAN),geometryColor);
  }
  
  OPTIX_INTERSECT_PROGRAM(SpheresIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<Spheres::DD>();
    auto &ray = owl::getPRD<Ray>();

    vec3f center = self.origins[primID];
    float radius = self.radii?self.radii[primID]:self.defaultRadius;

    // with "move the origin" trick; see Ray Tracing Gems 2
    const vec3f old_org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    vec3f org = old_org;
    float t_move = max(0.f,length(center - old_org)-3.f*radius);
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
    if (hit_t < t_max) {
      hit_t += t_move;

      // vec3f P = old_org + hit_t * dir;
      // vec3f N = normalize(P-center);
      // P = optixTransformPointFromObjectToWorldSpace(center + radius * N);
      // N = optixTransformNormalFromObjectToWorldSpace(N);
      // vec3f geometryColor = NAN;
      // if (self.colors)
      //   geometryColor = self.colors[primID];
      
      // ray.setHit(P,N,hit_t,self.material,vec2f(NAN),geometryColor);
      optixReportIntersection(hit_t, 0);
    }
  }
  
}
