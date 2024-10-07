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
  using namespace barney::render;

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
  
  OPTIX_BOUNDS_PROGRAM(CylindersBounds)(const void *geomData,
                                        owl::common::box3f &bounds,  
                                        const int32_t primID)
  {
    const Cylinders::DD &geom = *(const Cylinders::DD *)geomData;
    vec2i idx = geom.indices[primID];
    vec3f a = geom.vertices[idx.x];
    vec3f b = geom.vertices[idx.y];
    float ra, rb;
    if (geom.radiusPerVertex) {
      ra = geom.radii[idx.x];
      rb = geom.radii[idx.y];
    } else {
      ra = rb = geom.radii[primID];
    }
    box3f box_a = {a-ra,a+ra};
    box3f box_b = {b-rb,b+rb};
    bounds.lower = min(box_a.lower,box_b.lower);
    bounds.upper = max(box_a.upper,box_b.upper);
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
    Ray &ray    = getPRD<Ray>();

    const vec2i idx = self.indices[primID];
    const vec3f v0  = self.vertices[idx.x];
    const vec3f v1  = self.vertices[idx.y];


#if 1
    const float r0 
      = self.radiusPerVertex
      ? self.radii[idx.x]
      : self.radii[primID];
    const float r1 
      = self.radiusPerVertex
      ? self.radii[idx.y]
      : self.radii[primID];
    vec3f ray_org  = optixGetObjectRayOrigin();
    vec3f ray_dir  = optixGetObjectRayDirection();
    float len_dir = length(ray_dir);
    vec3f objectN;

    float t0 = 0.f, t1 = ray.tMax;
    box3f bb;
    bb.extend(v0+r0);
    bb.extend(v0-r0);
    bb.extend(v1+r1);
    bb.extend(v1-r1);
    if (!boxTest(ray_org,ray_dir,
                 t0,t1,
                 bb)) return;
    float t_move = .99*t0;
    
    // float d01 = length(v1-v0);
    
    // float d_v0 = length(v0-ray_org);
    // float d_v1 = length(v1-ray_org);
    // float t_move = max(0.5f*(d_v0+d_v1),max(r0,r1));
    
    // t_move = t_move * 1.f/len_dir;
    // t_move = max(0.f,min(t_move,tMax*.95f));

    // if (ray.dbg) printf("t_move %f\n",t_move);
    ray_org = ray_org + t_move * ray_dir;
    float hit_t = ray.tMax - t_move;
    
    if (!intersectRoundedCone(v0,v1,r0,r1,
                              ray_org,ray_dir,
                              hit_t,objectN))
      return;
    if (hit_t < 0.f || hit_t > ray.tMax-t_move) return;
    
    vec3f objectP = ray_org + hit_t * ray_dir;
    hit_t += t_move;

    float lerp_t = dot(objectP-v0,v1-v0)/(length(objectP-v0)*length(v1-v0));
    lerp_t = max(0.f,min(1.f,lerp_t));
    

    auto interpolator = [&](const GeometryAttribute::DD &attrib) -> float4
    { /* does not make sense for spheres *///return make_float4(0,0,0,1);
      const vec4f value_a = attrib.fromArray.valueAt(idx.x);
      const vec4f value_b = attrib.fromArray.valueAt(idx.y);
      const vec4f ret = (1.f-lerp_t)*value_a + lerp_t*value_b;
      // if (ray.dbg) printf("======================================================= lerp %f -> %f %f %f\n",lerp_t,ret.x,ret.y,ret.z);
      return ret;
    };

    render::HitAttributes hitData;//(OptixGlobals::get());
    hitData.objectPosition  = objectP;
    hitData.objectNormal    = normalize(objectN);
    hitData.worldPosition   = optixTransformPointFromObjectToWorldSpace(objectP);
    hitData.worldNormal     = normalize(optixTransformNormalFromObjectToWorldSpace(objectN));
    hitData.primID          = primID;
    hitData.t               = hit_t;

    float surfOfs_eps = 1.f;
    surfOfs_eps = max(surfOfs_eps,fabsf(hitData.worldPosition.x));
    surfOfs_eps = max(surfOfs_eps,fabsf(hitData.worldPosition.y));
    surfOfs_eps = max(surfOfs_eps,fabsf(hitData.worldPosition.z));
    surfOfs_eps *= 1e-5f;

    hitData.worldPosition += surfOfs_eps * hitData.worldNormal;
      
    // if (self.colors)
    //   if (self.colorPerVertex)
    //     //   (vec3f&)hitData.color = self.colors[primID];

    // if (ray.dbg)
    //   printf("capsule hit t %f pos %f %f %f nor %f %f %f\n",
    //          hit_t,
    //          hitData.objectPosition.x,
    //          hitData.objectPosition.y,
    //          hitData.objectPosition.z,
    //          hitData.objectNormal.x,
    //          hitData.objectNormal.y,
    //          hitData.objectNormal.z);
    self.setHitAttributes(hitData,interpolator,ray.dbg);

    const DeviceMaterial &material = OptixGlobals::get().materials[self.materialID];
    material.setHit(ray,hitData,OptixGlobals::get().samplers,ray.dbg);
    
    optixReportIntersection(hit_t, 0);
    
#else
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

    Ray &ray    = getPRD<Ray>();
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


    auto interpolator = [&](const GeometryAttribute::DD &attrib) -> float4
    { /* does not make sense for spheres *///return make_float4(0,0,0,1);

      // doesn't make sense, but anari sdk assumes for spheres per-vtx is same as per-prim
      float4 v = make_float4(0,0,0,1);//attrib.fromArray.valueAt(hitData.primID,ray.dbg);
      // if (ray.dbg)
      //   printf("querying attribute prim %i -> %f %f %f %f \n",hitData.primID,v.x,v.y,v.z,v.w);
      return v;
    };

    render::HitAttributes hitData;//(OptixGlobals::get());
    hitData.worldPosition   = optixTransformPointFromObjectToWorldSpace(objectP);
    hitData.objectPosition  = objectP;
    hitData.worldNormal     = objectN;
    hitData.objectNormal    = optixTransformNormalFromObjectToWorldSpace(objectN);
    hitData.primID          = primID;
    hitData.t               = t_hit;;
    // if (self.colors)
    //   (vec3f&)hitData.color = self.colors[primID];
    
    self.setHitAttributes(hitData,interpolator,ray.dbg);

    const DeviceMaterial &material = OptixGlobals::get().materials[self.materialID];
    material.setHit(ray,hitData,OptixGlobals::get().samplers,ray.dbg);
    
    optixReportIntersection(ray.tMax, 0);
#endif
  }
  
}
