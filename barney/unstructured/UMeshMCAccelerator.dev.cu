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

#include "barney/unstructured/UMeshMCAccelerator.h"
#include "owl/owl_device.h"

namespace barney {

  OPTIX_BOUNDS_PROGRAM(UMesh_MC_CUBQL_Bounds)(const void *geomData,                
                                       owl::common::box3f &bounds,  
                                       const int32_t primID)
  {
    auto self = *(const UMeshAccel_MC_CUBQL::DD*)geomData;
    vec3i dims = self.mcGrid.dims;
#if UMESH_MC_USE_DDA
    if (primID > 0)
      return;
    bounds = getBox(self.mesh.worldBounds);
#else
    if (primID >= dims.x*dims.y*dims.z)
      return;
    vec3i cellID(primID % dims.x,
                 (primID / dims.x) % dims.y,
                 primID / (dims.x*dims.y));

    // compute world-space bounds of macro cells
    bounds.lower
      = lerp(getBox(self.mesh.worldBounds),
             vec3f(cellID)*rcp(vec3f(dims)));
    bounds.upper
      = lerp(getBox(self.mesh.worldBounds),
             vec3f(cellID+vec3i(1))*rcp(vec3f(dims)));

    // printf("bounds (%f %f %f)(%f %f %f)\n",
    //        bounds.lower.x,
    //        bounds.lower.y,
    //        bounds.lower.z,
    //        bounds.upper.x,
    //        bounds.upper.y,
    //        bounds.upper.z);
    
# if 1
    // kill this prim is majorant is 0
    if (self.mcGrid.majorants &&
        self.mcGrid.majorants[primID] == 0.f)
      bounds = box3f();
# endif
#endif
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(UMesh_MC_CUBQL_CH)()
  {
    auto &ray = owl::getPRD<Ray>();
    const auto &self = getProgramData<UMeshAccel_MC_CUBQL::DD>();
    int primID = optixGetPrimitiveIndex();

    // ray.hadHit = true;
    // ray.color = .8f;//owl::randomColor(primID);
    ray.primID = primID;
    ray.tMax = optixGetRayTmax();
  }

  OPTIX_INTERSECT_PROGRAM(UMesh_MC_CUBQL_Isec)()
  {
    int primID = optixGetPrimitiveIndex();
    
    auto &ray = owl::getPRD<Ray>();
    const auto &self = getProgramData<UMeshAccel_MC_CUBQL::DD>();
    vec3i dims = self.mcGrid.dims;
    vec3i cellID(primID % dims.x,
                 (primID / dims.x) % dims.y,
                 primID / (dims.x*dims.y));

    // compute world-space bounds of macro cells
    box3f bounds;
    bounds.lower
      = lerp(getBox(self.mesh.worldBounds),
             vec3f(cellID)*rcp(vec3f(dims)));
    bounds.upper
      = lerp(getBox(self.mesh.worldBounds),
             vec3f(cellID+vec3i(1))*rcp(vec3f(dims)));
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };
    if (boxTest(ray,tRange,bounds))
      optixReportIntersection(tRange.upper, 0);
  }
}
