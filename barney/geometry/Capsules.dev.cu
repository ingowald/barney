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

#include "barney/geometry/Capsules.h"
#include "owl/owl_device.h"

namespace barney {
  using namespace barney::render;

  /* perform an actual ray-capsule intersection. Code is mostly stolen
     from inigo quielez' shadertoy example, and updated to fit this
     codebase. Note this code is not very numerically stable, so needs
     to be used with the "move-your-origin" trick (see ray tracing
     gems) to work in practice */
  inline __device__
  bool intersectRoundedCone(// first end-point
                            const vec4f a,
                            // second end-point
                            const vec4f b,
                            // ray origin
                            const vec3f ro,
                            // ray direction
                            const vec3f rd,
                            float& hit_t,
                            vec3f& isec_normal)
  {
    const float ra  = a.w;
    const float rb  = b.w;
    const vec3f pa  = getPos(a);
    const vec3f pb  = getPos(b);

#if 1
    float dist_ab = length(pa-pb);
    if (dist_ab+ra <= rb)
      return false;
    if (dist_ab+rb <= ra)
      return false;
#endif
    
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

    bool hadHit = false;
    
    float h = k1 * k1 - k0 * k2;
    if (h < 0.f) return false;
    float t = (-sqrtf(h) - k1) / k2;
    float y = m1 - ra * rr + t * m2;

    if (y > 0.f && y < d2 && t > 1e-6f && t < hit_t) {
      hit_t = t;
      isec_normal = normalize(d2 * (oa + t * rd) - ba * y);
      hadHit = true;
    }

    // // backside:
    // t = (+sqrtf(h) - k1) / k2;
    // y = m1 - ra * rr + t * m2;
    // if (y > 0.f && y < d2 && t < hit_t) {
    //   hit_t = t;
    //   isec_normal = normalize(d2 * (oa + t * rd) - ba * y);
    //   hadHit = true;
    // }

    // Caps. 
    float h1 = m3 * m3 - m5 + ra * ra;
    if (h1 > 0.f) {
      t = -m3 - sqrtf(h1);
      if (t > 1e-6f && t < hit_t) {
        hit_t = t;
        isec_normal = normalize((ro + t * rd - pa) / ra);
        hadHit = true;
        // isec_normal = normalize((oa + t * rd) / ra);
      }
    }
#if 1
    float h2 = m6 * m6 - m7 + rb * rb;
    if (h2 > 0.f) {
      t = -m6 - sqrtf(h2);
      if (t > 1e-6f && t < hit_t) {
        hit_t = t;
        isec_normal = normalize((ro + t * rd - pb) / rb);
        hadHit = true;
      }
    }
#endif
    return hadHit;
  }

  inline __device__ box3f getBounds(vec4f va, vec4f vb)
  {
    vec3f a  = getPos(va);
    vec3f b  = getPos(vb);
    float ra = va.w, rb = vb.w;
    box3f box_a = {a-ra,a+ra};
    box3f box_b = {b-rb,b+rb};
    return box3f(min(box_a.lower,box_b.lower),
                 max(box_a.upper,box_b.upper));
  }
  
  /*! boudns program for a single capsule, computes as bbox of the two
      end-cap spheres */
  OPTIX_BOUNDS_PROGRAM(CapsulesBounds)(const void *geomData,
                                       owl::common::box3f &bounds,  
                                       const int32_t primID)
  {
    const Capsules::DD &geom = *(const Capsules::DD *)geomData;
    vec2i indices = geom.indices[primID];
    bounds = getBounds(geom.vertices[indices.x],
                       geom.vertices[indices.y]);
  }

  /*! closest-hit program - doesn't do anything because we do all the
      work in IS prog, but needs to exist to make optix happy */
  OPTIX_CLOSEST_HIT_PROGRAM(CapsulesCH)()
  {
    /* nothing - already set in isec */
  }

  /*! optix intersection program for capsules geometry; will use the
      move-your-origin trick to improve numerical robustness and call
      quielez intersector. Unlike regular optix is programs this
      _will_ modify the ray and store the hit point if an intersection
      is found */
  OPTIX_INTERSECT_PROGRAM(CapsulesIsec)()
  {
    const int primID
      = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<Capsules::DD>();
    Ray &ray    = getPRD<Ray>();

    const vec2i idx = self.indices[primID];

    // get the end points and radii
    const vec4f v0 = self.vertices[idx.x];
    const vec4f v1 = self.vertices[idx.y];

    // ray from optix: this is an IS prog so this is _object_ space
    vec3f ray_org  = optixGetObjectRayOrigin();
    vec3f ray_dir  = optixGetObjectRayDirection();
    float len_dir = length(ray_dir);
    vec3f objectN;

    // set up move-your-origin trick: compute bbox, perform ray-box
    // test, and use the box enter distance as move-your-origin
    // distance
    float t0 = 0.f, t1 = ray.tMax;
    box3f bb = getBounds(v0,v1);
    if (!boxTest(ray_org,ray_dir,
                 t0,t1,
                 bb)) return;

#if 1
    render::HitAttributes hitData;
    const DeviceMaterial &material
      = OptixGlobals::get().materials[self.materialID];
    PackedBSDF bsdf
      = material.createBSDF(hitData,OptixGlobals::get().samplers,ray.dbg);
    float opacity
      = bsdf.getOpacity(ray.isShadowRay,ray.isInMedium,
                        ray.dir,hitData.worldNormal,ray.dbg);
    if (opacity < 1.f && ((Random &)ray.rngSeed)() < 1.f-opacity) {
      optixIgnoreIntersection();
      return;
    }
#endif
    
    
    
    // move just a little bit less in case the ray enters the box just
    // where it touches the prim
    float t_move = .99*t0;

    // move the origin forward
    ray_org = ray_org + t_move * ray_dir;
    float hit_t = ray.tMax - t_move;


    if (!intersectRoundedCone(v0,v1,
                              ray_org,ray_dir,
                              hit_t,objectN))
      // no intersection, ignore
      return;
    if (hit_t < 0.f || hit_t > ray.tMax-t_move)
      // intersection not in valid ray interval, ignore
      return;

    // compute object hit point _before_ moving back the distance -
    // this uses the forward-moved origin with the 'backward'-moved
    // distance, so is the valid position
    vec3f objectP = ray_org + hit_t * ray_dir;

    // this is the hit distance relative to the _original_ origin
    hit_t += t_move;

    // compute an interpolation weight for the anari-style attribute
    // interpolation
    const vec3f _v0 = getPos(v0);
    const vec3f _v1 = getPos(v1);
    float l01 = length(_v1-_v0);
    float lp0 = length(objectP-_v0);
    float lerp_t = 0.f;
    if (l01 < 1e-8f || lp0 < 1e-8f)
      lerp_t = 0.f;
    else {
      lerp_t
        = dot(objectP-_v0,_v1-_v0)
        / (lp0*l01);
      lerp_t = max(0.f,min(1.f,lerp_t));
    }

    // interpolator for anari-style color/attribute interpolation
    auto interpolator = [&](const GeometryAttribute::DD &attrib) -> float4
    {
      const vec4f value_a = attrib.fromArray.valueAt(idx.x);
      const vec4f value_b = attrib.fromArray.valueAt(idx.y);
      const vec4f ret = (1.f-lerp_t)*value_a + lerp_t*value_b;
      return ret;
    };

    // set up hit data for anari hit attributes
    hitData.primID          = primID;
    hitData.t               = hit_t;
    hitData.objectPosition  = objectP;
    hitData.objectNormal    = normalize(objectN);
    hitData.worldPosition
      = optixTransformPointFromObjectToWorldSpace(objectP);
    hitData.worldNormal
      = normalize(optixTransformNormalFromObjectToWorldSpace(objectN));

    // compute a stable epsilon for surface offsetting
    float surfOfs_eps = 1.f;
    surfOfs_eps = max(surfOfs_eps,fabsf(hitData.worldPosition.x));
    surfOfs_eps = max(surfOfs_eps,fabsf(hitData.worldPosition.y));
    surfOfs_eps = max(surfOfs_eps,fabsf(hitData.worldPosition.z));
    surfOfs_eps *= 1e-5f;

    // ... and shift the hit point just a little bit off the surface
    hitData.worldPosition += surfOfs_eps * hitData.worldNormal;

    // trigger the anari attribute evaluation
    self.setHitAttributes(hitData,interpolator,ray.dbg);

    // ... store the hit in the ray, rqs-style ...
    // const DeviceMaterial &material = OptixGlobals::get().materials[self.materialID];
    material.setHit(ray,hitData,OptixGlobals::get().samplers,ray.dbg);

    // .... and let optix know we did have a hit.
    optixReportIntersection(hit_t, 0);
  }
  
}
