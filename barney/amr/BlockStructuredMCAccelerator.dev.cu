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

#include "barney/amr/BlockStructuredMCAccelerator.h"
#include <owl/owl_device.h>

namespace barney {

  struct Woodcock {
    template<typename VolumeSampler>
    static inline __device__
    bool sampleRange(vec4f &sample,
                     const VolumeSampler &volume,
                     vec3f org, vec3f dir,
                     range1f &tRange,
                     float majorant,
                     uint32_t &rngSeed,
                     bool dbg=false)
    {
      LCG<4> &rand = (LCG<4> &)rngSeed;
      float t = tRange.lower;
      while (true) {
        float dt = - logf(1.f-rand())/majorant;
        t += dt;
        if (t >= tRange.upper)
          return false;
      
        sample = volume.sampleAndMap(org+t*dir,dbg);
        if (sample.w >= rand()*majorant) {
          tRange.upper = t;
          return true;
        }
      }
    }
  };
  
  OPTIX_BOUNDS_PROGRAM(BlockStructured_MC_CUBQL_Bounds)(const void *geomData,
                                                        owl::common::box3f &bounds,
                                                        const int32_t primID)
  {
    auto self = *(const BlockStructuredAccel_MC_CUBQL::DD*)geomData;
    vec3i dims = self.mcGrid.dims;
    const auto &field = self.sampler.field;
    if (primID >= dims.x*dims.y*dims.z)
      return;

    vec3i cellID(primID % dims.x,
                 (primID / dims.x) % dims.y,
                 primID / (dims.x*dims.y));

    // compute world-space bounds of macro cells
    bounds.lower
      = lerp(getBox(field.worldBounds),
             vec3f(cellID)*rcp(vec3f(dims)));
    bounds.upper
      = lerp(getBox(field.worldBounds),
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
  }

  OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MC_CUBQL_CH)()
  {
    auto &ray = owl::getPRD<Ray>();
    const auto &self = getProgramData<BlockStructuredAccel_MC_CUBQL::DD>();
    int primID = optixGetPrimitiveIndex();

    ray.tMax = optixGetRayTmax();

    vec3f P = ray.org + ray.tMax * ray.dir;
    vec4f mapped = self.sampleAndMap(P);

    ray.hadHit = true;
    ray.hit.baseColor
      = getPos(mapped)
      // * (.3f+.7f*fabsf(dot(normalize(ray.dir),N)))
      ;
    ray.hit.N = vec3f(0.f);
    ray.hit.P = P[6];
  }

  OPTIX_INTERSECT_PROGRAM(BlockStructured_MC_CUBQL_Isec)()
  {
    int primID = optixGetPrimitiveIndex();
    
    auto &ray = owl::getPRD<Ray>();
    const auto &self = getProgramData<BlockStructuredAccel_MC_CUBQL::DD>();
    const auto &field = self.sampler.field;
    vec3i dims = self.mcGrid.dims;
    vec3i cellID(primID % dims.x,
                 (primID / dims.x) % dims.y,
                 primID / (dims.x*dims.y));
    
    // compute world-space bounds of macro cells
    box3f bounds;
    bounds.lower
      = lerp(getBox(field.worldBounds),
             vec3f(cellID)*rcp(vec3f(dims)));
    bounds.upper
      = lerp(getBox(field.worldBounds),
             vec3f(cellID+vec3i(1))*rcp(vec3f(dims)));
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };
    
    if (!boxTest(ray,tRange,bounds))
      return;

    float majorant = self.mcGrid.majorants[primID];
    vec4f sample = 0.f;
    if (!Woodcock::sampleRange(sample,self,
                               ray.org,ray.dir,tRange,majorant,ray.rngSeed,
                               ray.dbg))
      return;

    ray.tMax = tRange.upper;
    optixReportIntersection(tRange.upper, 0);
  }
}
