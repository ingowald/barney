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

#include "barney/Triangles.h"
#include "owl/owl_device.h"

namespace barney {
  
  OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
  {
    auto &ray = owl::getPRD<Ray>();
    auto &self = owl::getProgramData<Triangles::DD>();
    ray.hadHit = true;
    ray.tMax = optixGetRayTmax();
    int primID = optixGetPrimitiveIndex();
    vec3i triangle = self.indices[primID];
    vec3f a = self.vertices[triangle.x];
    vec3f b = self.vertices[triangle.y];
    vec3f c = self.vertices[triangle.z];
    vec3f n = normalize(cross(b-a,c-a));
    vec3f dir = optixGetWorldRayDirection();
    vec3f baseColor = owl::randomColor(primID);
    ray.color = .3f + .7f*baseColor*abs(dot(dir,n));
    // printf("Marking ray %i as hit\n",ray.pixelID);
  }
  
}
