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
  {}


  /*! triangles geom AH program; mostly check on transparency */
  OPTIX_ANY_HIT_PROGRAM(TrianglesAH)()
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
    vec3f n = cross(v1-v0,v2-v0);
    // vec3f ws_n = soptixTransformNormalFromObjectToWorldSpace(n);
    if (0 && ray.dbg)
      printf("----------- Triangles::AH (%i %p) at %f\n",
             primID,&self,optixGetRayTmax());
    // if (0 && ray.dbg)
    //   printf("geom normal %f %f %f world %f %f %f\n",
    //          n.x,n.y,n.z,
    //          ws_n.x,ws_n.y,ws_n.z);
    if (self.normals) {
      vec3f Ns
        = (1.f-u-v) * self.normals[triangle.x]
        + (    u  ) * self.normals[triangle.y]
        + (      v) * self.normals[triangle.z];
      Ns = normalize(Ns);

      // vec3f ws_Ns
      //   = optixTransformNormalFromObjectToWorldSpace(Ns);
      // if (0 && ray.dbg)
      //   printf("shading normal %f %f %f world %f %f %f\n",
      //          Ns.x,Ns.y,Ns.z,
      //          ws_Ns.x,
      //          ws_Ns.y,
      //          ws_Ns.z
      //          );

      if (dot(Ns,(vec3f)optixGetObjectRayDirection()) > 0.f)
        Ns = n;
        
      // if (dot(n,(vec3f)optixGetObjectRayDirection()) < 0.f) {
      //   if (dot(Ns,n) < 0.f)
      //     Ns = reflect(Ns,n);
      // } else {
      //   if (dot(Ns,n) > 0.f)
      //     Ns = reflect(Ns,n);
      // }
      n = Ns;
    }
    if (0 && ray.dbg)
      printf("final normal %f %f %f\n",
             n.x,n.y,n.z);
      //   else 
      // n = cross(v1-v0,v2-v0);

    const vec3f osN = normalize(n);
    n = optixTransformNormalFromObjectToWorldSpace(n);
    n = normalize(n);
    
    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
    vec3f P  = optixTransformPointFromObjectToWorldSpace(osP);

    render::HitAttributes hitData;
    hitData.worldPosition   = P;
    hitData.worldNormal     = n;
    hitData.objectPosition  = osP;
    hitData.objectNormal    = osN;
    hitData.primID          = primID;
    hitData.t               = optixGetRayTmax();
    hitData.isShadowRay     = ray.isShadowRay;
    
    auto interpolator
      = [&](const GeometryAttribute::DD &attrib) -> float4
      {
        const vec4f value_a = attrib.fromArray.valueAt(triangle.x);
        const vec4f value_b = attrib.fromArray.valueAt(triangle.y);
        const vec4f value_c = attrib.fromArray.valueAt(triangle.z);
        const vec4f ret = (1.f-u-v)*value_a + u*value_b + v*value_c;
        return ret;
      };
    self.setHitAttributes(hitData,interpolator,ray.dbg);

    const DeviceMaterial &material
      = OptixGlobals::get().materials[self.materialID];
    PackedBSDF bsdf
      = material.createBSDF(hitData,OptixGlobals::get().samplers,ray.dbg);
#if 0
    float opacity
      = material.getOpacity(hitData,OptixGlobals::get().samplers,ray.dbg);
#else
    float opacity
      = bsdf.getOpacity(ray.isShadowRay,ray.isInMedium,
                        ray.dir,hitData.worldNormal,ray.dbg);
#endif

    if (opacity < 1.f && ((Random &)ray.rngSeed)() < 1.f-opacity) {
      optixIgnoreIntersection();
      return;
    }
    else {
      material.setHit(ray,hitData,OptixGlobals::get().samplers,ray.dbg);
    }
  }
  
}
