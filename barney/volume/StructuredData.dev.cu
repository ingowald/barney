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

#include "barney/volume/StructuredData.h"
#include <owl/owl_device.h>

namespace barney {

  OPTIX_BOUNDS_PROGRAM(Structured_MCRTX_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    // "RTX": we need to create one prim per macro cell
    
    // for now, do not use refitting, simple do rebuild every
    // frame... in this case we can simply return empty box for every
    // inactive cell.

    using Geom = typename MCRTXVolumeAccel<StructuredData>::DD;
    const Geom &self = *(Geom*)geomData;
    if (primID >= self.mcGrid.numCells()) return;
    
    const float maj = self.mcGrid.majorants[primID];
    if (maj == 0.f) {
      bounds = box3f();
    } else {
      const vec3i mcID = self.mcGrid.cellID(primID);
      bounds = self.mcGrid.cellBounds(mcID,self.worldBounds);
      // vec3f rel_lo = vec3f(mcID) * rcp(vec3f(self.mcGrid.dims));
      // vec3f rel_hi = vec3f(mcID+vec3i(1)) * rcp(vec3f(self.mcGrid.dims));
      // bounds.lower = lerp(self.worldBounds,rel_lo);
      // bounds.upper = lerp(self.worldBounds,rel_hi);
      // printf("active cell %f %f %f\n",
      //        bounds.lower.x,bounds.lower.y,bounds.lower.z);
    }
  }

  OPTIX_INTERSECT_PROGRAM(Structured_MCRTX_Isec)()
  {
    using Geom = typename MCRTXVolumeAccel<StructuredData>::DD;
    const Geom &self = owl::getProgramData<Geom>();
    Ray &ray = owl::getPRD<Ray>();
    const int primID = optixGetPrimitiveIndex();

    bool dbg = ray.dbg;
    
    const vec3i mcID = self.mcGrid.cellID(primID);
    
    const float majorant = self.mcGrid.majorants[primID];
    if (dbg) printf("isec mc ID %i %i %i maj %f\n",mcID.x,mcID.y,mcID.z,majorant);
    if (majorant == 0.f) return;
    
    box3f bounds = self.mcGrid.cellBounds(mcID,self.worldBounds);
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };

    if (!boxTest(ray,tRange,bounds))
      return;
    if (dbg) printf("range %f %f\n",tRange.lower,tRange.upper);
    
    vec4f sample = 0.f;
    if (!Woodcock::sampleRange(sample,self,
                               ray.org,ray.dir,tRange,majorant,ray.rngSeed,
                               ray.dbg))
      return;

    ray.tMax = tRange.upper;
    ray.hit.baseColor = getPos(sample);
    ray.hit.N = vec3f(0.f);
    ray.hit.P = ray.org + tRange.upper*ray.dir;
    optixReportIntersection(tRange.upper, 0);
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(Structured_MCRTX_CH)()
  {
    /* nothing - already all set in isec */
  }
  
}

