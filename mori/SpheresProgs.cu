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

#include "mori/Spheres.h"
#include "owl/owl_device.h"

namespace mori {
  
  OPTIX_BOUNDS_PROGRAM(SphereBounds)(const void *geomData,                
                                     owl::common::box3f &bounds,  
                                     const int32_t primID)
  {
    const Spheres::OnDev &geom = *(const Spheres::OnDev *)geomData;
    vec3f origin = geom.origins[primID];
    bounds.lower = origin - geom.defaultRadius;
    bounds.upper = origin + geom.defaultRadius;
  }

  OPTIX_INTERSECT_PROGRAM(SphereIsec)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<Spheres::OnDev>();

    vec3f center = self.origins[primID];
    float radius = self.defaultRadius;
    
    /* iw, jan 11, 2020: for this particular example (where we do not
       yet use instancing) we could also use the World ray; but this
       version is cleaner since it would/will also work with
       instancing */
    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float hit_t      = optixGetRayTmax();
    const float tmin = optixGetRayTmin();
    
    const vec3f oc = org - center;
    const float a = dot(dir,dir);
    const float b = dot(oc, dir);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - a * c;
    
    if (discriminant < 0.f) return;

    {
      float temp = (-b - sqrtf(discriminant)) / a;
      if (temp < hit_t && temp > tmin) 
        hit_t = temp;
    }
      
    {
      float temp = (-b + sqrtf(discriminant)) / a;
      if (temp < hit_t && temp > tmin) 
        hit_t = temp;
    }
    if (hit_t < optixGetRayTmax()) {
      optixReportIntersection(hit_t, 0);
    }
  }
  
}
