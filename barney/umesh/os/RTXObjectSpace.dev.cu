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

#include "barney/umesh/os/RTXObjectSpace.h"
#include <owl/owl_device.h>

namespace barney {

  using Cluster = RTXObjectSpace::Cluster;

  OPTIX_BOUNDS_PROGRAM(RTXObjectSpaceBounds)(const void *geomData,                
                                             owl::common::box3f &primBounds,  
                                             const int32_t primID)
  {
    const auto &self = *(const RTXObjectSpace::DD *)geomData;
    Cluster &cluster = self.clusters[primID];
    int begin = self.clusters[primID].begin;
    int end   = self.clusters[primID].end;

    if (self.firstTimeBuild) {
      // cluster bounds not yet set - this must be a first-time build
      // - let's set them.
      cluster.bounds = box4f();
      for (int i=begin;i<end;i++)
        cluster.bounds.extend(self.eltBounds(self.elements[i]));
      primBounds = getBox(cluster.bounds);
    } else {
      box4f bounds  = cluster.bounds;
         // printf("cluster.bounds  (%f %f %f)(%f %f %f)\n",
         //        bounds.lower.x,
         //        bounds.lower.y,
         //        bounds.lower.z,
         //        bounds.upper.x,
         //        bounds.upper.y,
         //        bounds.upper.z);
      range1f range = getRange(bounds);
      float majorant = self.xf.majorant(range);
      cluster.majorant = majorant;
      if (majorant == 0.f) {
        // swap(primBounds.lower,primBounds.upper);
        primBounds = box3f();
        // printf("bounds prog %i -> CULLED\n",primID);
      } else {
        primBounds = getBox(bounds);
         // printf("bounds prog %i -> ALIVE (%f %f %f)(%f %f %f)\n",primID,
         //        primBounds.lower.x,
         //        primBounds.lower.y,
         //        primBounds.lower.z,
         //        primBounds.upper.x,
         //        primBounds.upper.y,
         //        primBounds.upper.z);
      }
    }
  }

  OPTIX_CLOSEST_HIT_PROGRAM(RTXObjectSpaceCH)()
  {
    // auto &ray = owl::getPRD<Ray>();
    // auto &self = owl::getProgramData<RTXObjectSpace::DD>();
    // int primID = optixGetPrimitiveIndex();
    // // Cluster &cluster = self.clusters[primID];
    // // int begin = cluster.begin;
    // // int end = cluster.end;
    // // float majorant = cluster.majorant;
    
    // ray.tMax = optixGetRayTmax();

    // vec3f P = ray.org + ray.tMax * ray.dir;

    // // CentralDifference cd(self,self.xf,P,begin,end,ray.dbg);

    // // vec3f N = normalize
    // //   ((cd.N == vec3f(0.f)) ? ray.dir : cd.N);
    // ray.hadHit = 1;
    // ray.hit.N = vec3f(0.f);
    // ray.hit.P = P;
    // // ray.hit.baseColor = cd.mappedColor;
  }



  OPTIX_INTERSECT_PROGRAM(RTXObjectSpaceIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<typename RTXObjectSpace::DD>();
    auto &ray
      = owl::getPRD<Ray>();
    // bool dbg = ray.dbg;

// #if PRINT_BALLOT
//     int numActive = __popc(__ballot(1));
//     if (ray.dbg)
//       printf("### isec on geom %lx, leaf %i, numActive = %i\n",(void *)&self,primID,numActive);
// #endif
    
    Cluster cluster = self.clusters[primID];
    int begin = cluster.begin;
    int end = cluster.end;
    // float majorant = cluster.majorant;
    box3f bounds = getBox(cluster.bounds);

    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    // if (ray.dbg) printf("ray range %f %f\n",t0,t1);
    bool isHittingTheBox
      = boxTest(t0,t1,bounds,org,dir);
    if (!isHittingTheBox) 
      return;

    range1f leafRange(t0,t1);
    // if (ray.dbg) printf("---------------------- leaf range %f %f\n",
    //                     leafRange.lower,leafRange.upper);

    vec3f sample;
    float hit_t = intersectLeaf(ray,leafRange,self,begin,end,0);//,ray.dbg);
    if (hit_t < optixGetRayTmax())
      optixReportIntersection(hit_t, 0);
  }

}
