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

#include "barney/geometry/Triangles.h"
#include <owl/owl_device.h>

namespace barney {
  
  OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Triangles::DD>();
    ray.hadHit = true;
    ray.tMax = optixGetRayTmax();
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    vec3f v0 = self.vertices[triangle.x];
    vec3f v1 = self.vertices[triangle.y];
    vec3f v2 = self.vertices[triangle.z];
    vec3f n = cross(v1-v0,v2-v0);
    n = optixTransformNormalFromObjectToWorldSpace(n);
    n = normalize(n);
    
    vec3f dir = optixGetWorldRayDirection();
#if VISUALIZE_PRIMS
    vec3f baseColor = owl::randomColor(primID);
#else
    vec3f baseColor = self.material.baseColor;
#endif
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    
    const vec3f osP  = (1.f-u-v)*v0 + u*v1 + v*v2;
    vec3f P  = optixTransformPointFromObjectToWorldSpace(osP);
    
    ray.hit.baseColor = baseColor;
    ray.hit.N         = n;
    ray.hit.P         = P;
  }
  
}
