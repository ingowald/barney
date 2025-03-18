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
#include "rtcore/TraceInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;
    
  struct TrianglesPrograms {
      
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &rt)
    {
      printf("CLOSEST!\n");
    }

    static inline __rtc_device
    void anyHit(rtc::TraceInterface &rt)
    {
      auto &ray = *(Ray *)rt.getPRD();
      
#if NDEBUG
      bool dbg = false;
#else
      bool dbg = ray.dbg;
#endif
      
      auto &self = *(Triangles::DD*)rt.getProgramData();
      const float u = rt.getTriangleBarycentrics().x;
      const float v = rt.getTriangleBarycentrics().y;
      int primID = rt.getPrimitiveIndex();
      vec3i triangle = self.indices[primID];
      vec3f v0 = self.vertices[triangle.x];
      vec3f v1 = self.vertices[triangle.y];
      vec3f v2 = self.vertices[triangle.z];
      vec3f n = cross(v1-v0,v2-v0);
      if (self.normals) {
        vec3f Ns
          = (1.f-u-v) * self.normals[triangle.x]
          + (    u  ) * self.normals[triangle.y]
          + (      v) * self.normals[triangle.z];
        Ns = normalize(Ns);

        if (dot(Ns,(vec3f)rt.getObjectRayDirection()) > 0.f)
          Ns = n;
        
        n = Ns;
      }
      const vec3f osN = normalize(n);
      n = rt.transformNormalFromObjectToWorldSpace(n);
      n = normalize(n);

      // ------------------------------------------------------------------
      // get texture coordinates
      // ------------------------------------------------------------------
      const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
      vec3f P  = rt.transformPointFromObjectToWorldSpace(osP);

      render::HitAttributes hitData;
      hitData.worldPosition   = P;
      hitData.worldNormal     = n;
      hitData.objectPosition  = osP;
      hitData.objectNormal    = osN;
      hitData.primID          = primID;
      hitData.t               = rt.getRayTmax();
      hitData.isShadowRay     = ray.isShadowRay;

      auto interpolator
        = [&](const GeometryAttribute::DD &attrib) -> vec4f
        {
          const vec4f value_a = attrib.fromArray.valueAt(triangle.x,dbg);
          const vec4f value_b = attrib.fromArray.valueAt(triangle.y,dbg);
          const vec4f value_c = attrib.fromArray.valueAt(triangle.z,dbg);
          const vec4f ret = (1.f-u-v)*value_a + u*value_b + v*value_c;
          return ret;
        };
      self.setHitAttributes(hitData,interpolator,dbg);

      const DeviceMaterial &material
        = OptixGlobals::get(rt).materials[self.materialID];
      PackedBSDF bsdf
        = material.createBSDF(hitData,OptixGlobals::get(rt).samplers,dbg);
      float opacity
        = bsdf.getOpacity(ray.isShadowRay,ray.isInMedium,
                          ray.dir,hitData.worldNormal,ray.dbg);
      if (opacity < 1.f && ((Random &)ray.rngSeed)() < 1.f-opacity) {
        rt.ignoreIntersection();
        return;
      }
      else {
        material.setHit(ray,hitData,OptixGlobals::get(rt).samplers,ray.dbg);
      }
    }
  };

  RTC_EXPORT_TRIANGLES_GEOM(Triangles,Triangles::DD,TrianglesPrograms,true,false);
}

