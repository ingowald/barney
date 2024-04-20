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
    // vec3f Ng = cross(v1-v0,v2-v0);
    vec3f n;
    if (self.normals) {
      n
        = (1.f-u-v) * self.normals[triangle.x]
        + (    u  ) * self.normals[triangle.y]
        + (      v) * self.normals[triangle.z];
      // if (dot(n,Ng) < 0.f) n = -n;
    } else 
      n = cross(v1-v0,v2-v0);
    
    // if (ray.dbg) {
    //   printf("normals %i\n %f %f %f -> %f %f %f\n",int(self.normals),
    //          Ng.x,Ng.y,Ng.z,n.x,n.y,n.z);
      
    // }
    
    n = optixTransformNormalFromObjectToWorldSpace(n);
    n = normalize(n);
    
    vec3f dir = optixGetWorldRayDirection();
    // auto mat = self.material;



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
#if VISUALIZE_PRIMS
    colorFromTexture /*mat.baseColor*/ *= owl::randomColor(primID);
#endif

#if 1
    vec3f geometryColor(getColor(self,primID,triangle,u,v));
#endif

    if (ray.dbg)
      printf("setting hit geomcolor %f %f %f\n",
             geometryColor.x,
             geometryColor.y,
             geometryColor.z);
    const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
    vec3f P  = optixTransformPointFromObjectToWorldSpace(osP);
    ray.setHit(P,n,optixGetRayTmax(),
               self.material,tc,geometryColor);
  }



  /*! triangles geom AH program; mostly check on transparency */
  OPTIX_ANY_HIT_PROGRAM(TrianglesAH)()
  {
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
  }
  
}
