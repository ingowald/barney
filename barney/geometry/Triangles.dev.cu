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
#include "rtcore/ProgramInterface.h"

RTC_DECLARE_GLOBALS(BARNEY_NS::render::OptixGlobals);

namespace BARNEY_NS {
  using namespace BARNEY_NS::render;
    
  struct TrianglesPrograms {
#if RTC_DEVICE_CODE
    static inline __rtc_device
    void closestHit(rtc::TraceInterface &rt)
    {}

    static inline __rtc_device
    void anyHit(rtc::TraceInterface &ti)
    {
      auto &ray = *(Ray *)ti.getPRD();

      const OptixGlobals &globals = OptixGlobals::get(ti);
      const World::DD &world = globals.world;
#ifdef NDEBUG
      bool dbg = false;
#else
      bool dbg = ray.dbg;
#endif
      auto &self = *(Triangles::DD*)ti.getProgramData();
      const float u = ti.getTriangleBarycentrics().x;
      const float v = ti.getTriangleBarycentrics().y;
      int primID    = ti.getPrimitiveIndex();
      int instID    = ti.getInstanceID();
      float depth   = ti.getRayTmax();

      // ------------------------------------------------------------------
      if (globals.hitIDs) {
        /* ID buffer rendering writes IDs no matter what transparency */
        const int rayID
          = ti.getLaunchIndex().x
          + ti.getLaunchDims().x
          * ti.getLaunchIndex().y;
        if (depth < globals.hitIDs[rayID].depth) {
          globals.hitIDs[rayID].primID = primID;
          globals.hitIDs[rayID].instID
            = world.instIDToUserInstID
            ? world.instIDToUserInstID[instID]
            : instID;
          globals.hitIDs[rayID].objID  = self.userID;
          globals.hitIDs[rayID].depth  = depth;
        }
      }
      // ------------------------------------------------------------------

      
      vec3i triangle = self.indices[primID];
      vec3f v0 = self.vertices[triangle.x];
      vec3f v1 = self.vertices[triangle.y];
      vec3f v2 = self.vertices[triangle.z];
      vec3f n = normalize(cross(v1-v0,v2-v0));
      if (self.normals) {
        vec3f Ns
          = (1.f-u-v) * self.normals[triangle.x]
          + (    u  ) * self.normals[triangle.y]
          + (      v) * self.normals[triangle.z];
        Ns = normalize(Ns);
        n = Ns;
      }
      const vec3f osN = normalize(n);
      n = ti.transformNormalFromObjectToWorldSpace(n);
      n = normalize(n);

      // ------------------------------------------------------------------
      // get texture coordinates
      // ------------------------------------------------------------------
      const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
      vec3f P  = ti.transformPointFromObjectToWorldSpace(osP);

      render::HitAttributes hitData;
      hitData.worldPosition   = P;
      hitData.worldNormal     = n;
      hitData.objectPosition  = osP;
      hitData.objectNormal    = osN;
      hitData.primID          = primID;
      hitData.instID          = instID;
      hitData.t               = depth;
      hitData.isShadowRay     = ray.isShadowRay;

      auto interpolator
        = [triangle,u,v,dbg](const GeometryAttribute::DD &attrib) -> vec4f
        {
          const vec4f value_a = attrib.fromArray.valueAt(triangle.x,dbg);
          const vec4f value_b = attrib.fromArray.valueAt(triangle.y,dbg);
          const vec4f value_c = attrib.fromArray.valueAt(triangle.z,dbg);
          const vec4f ret = (1.f-u-v)*value_a + u*value_b + v*value_c;
          return ret;
        };
      self.setHitAttributes(hitData,interpolator,world,dbg);

      // if (dbg) printf("matid %i, world mat %lx\n",self.materialID,world.materials); 
      const DeviceMaterial &material
        = world.materials[self.materialID];
      
      // if (dbg) printf("creating bsdf\n");  
      // if (dbg) printf("creating bsdf type %i\n",int(material.type));  
      PackedBSDF bsdf
        = material.createBSDF(hitData,world.samplers,dbg);
// #if 1
//       material.setHit(ray,hitData,world.samplers,dbg);
// #else
      float opacity
        = bsdf.getOpacity(ray.isShadowRay,ray.isInMedium,
                          ray.dir,hitData.worldNormal,ray.dbg);
      
      if (opacity < 1.f) {
        int rayID = ti.getLaunchIndex().x+ti.getLaunchDims().x*ti.getLaunchIndex().y;
        Random rng(hash(rayID,
                        ti.getRTCInstanceIndex(),
                        ti.getGeometryIndex(),
                        ti.getPrimitiveIndex(),
                        world.rngSeed));
        if (rng() > opacity) {
          ti.ignoreIntersection();
          return;
        }
      }
      material.setHit(ray,hitData,world.samplers,dbg);
    }
#endif
  };
  
  RTC_EXPORT_TRIANGLES_GEOM(Triangles,Triangles::DD,TrianglesPrograms,true,false);
}

