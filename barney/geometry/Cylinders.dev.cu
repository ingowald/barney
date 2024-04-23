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

namespace barney {
  
  OPTIX_BOUNDS_PROGRAM(CylindersBounds)(const void *geomData,
                                        owl::common::box3f &bounds,  
                                        const int32_t primID)
  {
    const Cylinders::DD &geom = *(const Cylinders::DD *)geomData;
    vec2i idx = geom.indices[primID];
    float r   = geom.radii[primID];
    vec3f a = geom.vertices[idx.x];
    vec3f b = geom.vertices[idx.y];
    bounds.lower = min(a,b)-r;
    bounds.upper = max(a,b)+r;
  }

  OPTIX_CLOSEST_HIT_PROGRAM(CylindersCH)()
  {
    /* nothing - already set in isec */
  }
  
  OPTIX_INTERSECT_PROGRAM(CylindersIsec)()
  {
    // capped
    const int primID
      = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<Cylinders::DD>();

    const vec2i idx = self.indices[primID];
    const vec3f v0  = self.vertices[idx.x];
    const vec3f v1  = self.vertices[idx.y];
    const float radius
      = self.radiusPerVertex
      ? min(self.radii[idx.x],self.radii[idx.y])
      : self.radii[primID];

    const vec3f ray_org  = optixGetObjectRayOrigin();
    const vec3f ray_dir  = optixGetObjectRayDirection();
    float hit_t      = optixGetRayTmax();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    const vec3f d = ray_dir;
    const vec3f s = v1 - v0; // axis
    const vec3f sxd = cross(s, d);
    const float a = dot(sxd, sxd); // (s x d)^2
    if (a == 0.f)
      return;

    const vec3f f = v0 - ray_org;
    const vec3f sxf = cross(s, f);
    const float ra = 1.0f/a;
    const float ts = dot(sxd, sxf) * ra; // (s x d)(s x f) / (s x d)^2, in ray-space
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

    Ray &ray    = getPRD<Ray>();
    vec3f N = 0.f;
    if (ray_tmin <= t0 && t0 <= ray_tmax) {
      // front side hit:
      ray.tMax = t0;
      td *= -1.f;
      float hit_surf_u = (ray.tMax * sd - sf) * 1.f/(s2);
      N
        = (t0 == cap_t0)
        ? s
        : (td * d - fp - hit_surf_u * s);
      
    } else if (ray_tmin <= t1 && t1 <= ray_tmax) {
      ray.tMax = t1;
      float hit_surf_u = (ray.tMax * sd - sf) * 1.f/(s2);
      N
        = (t1 == cap_t1)
        ? -s
        : (td * d - fp - hit_surf_u * s);
    } else
      return;

    vec3f P = ray_org + ray.tMax * ray_dir;

    // THIS IS WRONG: !!!!!!!!!
    if (ray.dbg) printf("storing wrong normals here!\n");
    render::HitAttributes hitData(OptixGlobals::get());
    hitData.worldPosition   = P;
    hitData.objectPosition  = P;
    hitData.worldNormal     = N;
    hitData.objectNormal    = N;
    hitData.primID          = primID;
    hitData.t               = ray.tMax;
    // if (self.colors)
    //   (vec3f&)hitData.color = self.colors[primID];
    
    auto interpolate = [&](const render::GeometryAttribute::DD &)
    { /* does not make sense for spheres */return make_float4(0,0,0,1); };
    self.evalAttributesAndStoreHit(ray,hitData,interpolate);

    // ray.setHit(P,N,ray.tMax,self.material);
    
    optixReportIntersection(ray.tMax, 0);
  }
  
}
