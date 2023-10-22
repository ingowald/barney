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

#include "barney/unstructured/QuickClusters.h"
#include "owl/owl_device.h"

namespace barney {

  inline __device__
  bool boxTest(float &t0, float &t1,
               box3f box,
               const vec3f org,
               const vec3f dir)
  {
    vec3f t_lo = (box.lower - org) * rcp(dir);
    vec3f t_hi = (box.upper - org) * rcp(dir);
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);
    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    return t0 < t1;
  }

  inline __device__
  bool boxTest(float &t0, float &t1,
               box4f box,
               const vec3f org,
               const vec3f dir)
  {
    return boxTest(t0,t1,box3f({box.lower.x,box.lower.y,box.lower.z},
                               {box.upper.x,box.upper.y,box.upper.z}),
                   org,dir);
  }
  
  OPTIX_BOUNDS_PROGRAM(UMeshQCBounds)(const void *geomData,                
                                     owl::common::box3f &bounds,  
                                     const int32_t primID)
  {
    const auto &self = *(const UMeshQC::DD *)geomData;

    box4f clusterBounds;
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.numElements);
    for (int i=begin;i<end;i++) {
      UMeshQC::Element elt = self.elements[i];
      clusterBounds.extend(self.getBounds(elt));
    }

    bounds.lower = (const vec3f&)clusterBounds.lower;
    bounds.upper = (const vec3f&)clusterBounds.upper;

    if (length(bounds.span())>30) {
    printf("bounds %f %f %f : %f %f %f\n",
           bounds.lower.x,
           bounds.lower.y,
           bounds.lower.z,
           bounds.upper.x,
           bounds.upper.y,
           bounds.upper.z);
    }
  }

  OPTIX_CLOSEST_HIT_PROGRAM(UMeshQCCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<UMeshQC::DD>();
    int primID = optixGetPrimitiveIndex();

    ray.hadHit = true;
    ray.color = owl::randomColor(primID);
    ray.tMax = optixGetRayTmax();

  }
  
  OPTIX_INTERSECT_PROGRAM(UMeshQCIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<UMeshQC::DD>();
    
    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float ray_t0     = optixGetRayTmin();
    float ray_t1     = optixGetRayTmax();

    float hit_t = INFINITY;
    
    int begin = primID * UMeshQC::clusterSize;
    int end   = min(begin+UMeshQC::clusterSize,self.numElements);
    for (int eid=begin;eid<end;eid++) {
      UMeshQC::Element elt = self.elements[eid];
      box4f eltBounds = self.getBounds(elt);

      // printf("isec %f %f %f:%f %f %f box %f %f %f : %f %f %f\n",
      //        org.x,
      //        org.y,
      //        org.z,
      //        dir.x,
      //        dir.y,
      //        dir.z,
      //        eltBounds.lower.x,
      //        eltBounds.lower.y,
      //        eltBounds.lower.z,
      //        eltBounds.upper.x,
      //        eltBounds.upper.y,
      //        eltBounds.upper.z);
      
      float t0 = ray_t0;
      float t1 = min(ray_t1,hit_t);
      if (boxTest(t0,t1,eltBounds,org,dir)) {
        hit_t = t0;
      }
    }

    
    auto &ray = owl::getPRD<Ray>();
    Random rng(ray.rngSeed++,primID);

    // if ((primID % 16) == 0) {
    if (hit_t < 1e10f) {
      // printf("hit at %f\n",hit_t);
      optixReportIntersection(hit_t, 0);
    }
    // if (hit_t < optixGetRayTmax()) {
    //   optixReportIntersection(hit_t, 0);
    // }
  }
  
}
