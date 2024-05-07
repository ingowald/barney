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

#include "barney/geometry/Attributes.dev.h"
#include "barney/geometry/Triangles.h"
#include <owl/owl_device.h>

namespace barney {
  using namespace barney::render;
    
  OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Triangles::DD>();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    vec3f v0 = self.vertices[triangle.x];
    vec3f v1 = self.vertices[triangle.y];
    vec3f v2 = self.vertices[triangle.z];
    vec3f n;
    if (self.normals) {
      n
        = (1.f-u-v) * self.normals[triangle.x]
        + (    u  ) * self.normals[triangle.y]
        + (      v) * self.normals[triangle.z];
    } else 
      n = cross(v1-v0,v2-v0);

    const vec3f osN = n;
    n = optixTransformNormalFromObjectToWorldSpace(n);
    n = normalize(n);
    
    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    vec2f tc(u,v);
    if (self.texcoords) {
      const vec2f Ta = self.texcoords[triangle.x];
      const vec2f Tb = self.texcoords[triangle.y];
      const vec2f Tc = self.texcoords[triangle.z];
      tc = ((1.f-u-v)*Ta + u*Tb + v*Tc);
    }
    // #if 1
    //     vec3f geometryColor(getColor(self,primID,triangle,u,v));
    // #endif
    const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
    vec3f P  = optixTransformPointFromObjectToWorldSpace(osP);

    render::HitAttributes hitData;
    hitData.worldPosition   = P;
    hitData.worldNormal     = n;
    hitData.objectPosition  = osP;
    hitData.objectNormal    = osN;
    hitData.primID          = primID;
    hitData.t               = optixGetRayTmax();
    hitData.attribute[0]    = make_float4(tc.x,tc.y,0.f,1.f);

    // if (!isnan(geometryColor.x))
    //   (vec3f&)hitData.color = geometryColor;

    
    if (ray.dbg) printf("=== HIT TRIS setting hit attributes\n");
    auto interpolator
      = [&](const GeometryAttribute::DD &attrib) -> float4
      {
        const vec4f value_a = attrib.fromArray.valueAt(triangle.x);
        const vec4f value_b = attrib.fromArray.valueAt(triangle.y);
        const vec4f value_c = attrib.fromArray.valueAt(triangle.z);
        return (1.f-u-v)*value_a + u*value_b + v*value_c;
      };
    self.setHitAttributes(hitData,interpolator,ray.dbg);

    const DeviceMaterial &material = OptixGlobals::get().materials[self.materialID];
    if (ray.dbg) printf("=== HIT TRIS matID %i\n",self.materialID);
    material.setHit(ray,hitData,OptixGlobals::get().samplers,ray.dbg);
    // self.evalAttributesAndStoreHit(ray,hitData,interpolateAttrib);
    // ray.setHit(P,n,optixGetRayTmax(),
    //            self.material,tc,geometryColor);
  }



  /*! triangles geom AH program; mostly check on transparency */
  OPTIX_ANY_HIT_PROGRAM(TrianglesAH)()
  {
#if 0
    auto &ray  = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Triangles::DD>();

    if (!self.material.hasAlpha(ray.isShadowRay))
      return;
    
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    vec2f tc(u,v);
    if (self.texcoords) {
      const vec2f Ta = self.texcoords[triangle.x];
      const vec2f Tb = self.texcoords[triangle.y];
      const vec2f Tc = self.texcoords[triangle.z];
      tc = ((1.f-u-v)*Ta + u*Tb + v*Tc);
    }

    float alpha = self.material.getAlpha(tc,ray.isShadowRay);
    if (alpha < 1.f && ((Random &)ray.rngSeed)() < 1.f-alpha) {
      optixIgnoreIntersection();
      return;
    }
#endif
  }
  
}
