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
  
  OPTIX_BOUNDS_PROGRAM(UMesh_MC_CUBQL_Bounds)(const void *geomData,                
                                              owl::common::box3f &bounds,  
                                              const int32_t primID)
  {
    auto self = *(const UMeshAccel_MC_CUBQL::DD*)geomData;
    vec3i dims = self.mcGrid.dims;
    const auto &mesh = self.sampler.mesh;
#if UMESH_MC_USE_DDA
    if (primID > 0)
      return;
    bounds = getBox(mesh.worldBounds);
#else
    if (primID >= dims.x*dims.y*dims.z)
      return;
    vec3i cellID(primID % dims.x,
                 (primID / dims.x) % dims.y,
                 primID / (dims.x*dims.y));

    // compute world-space bounds of macro cells
    bounds.lower
      = lerp(getBox(mesh.worldBounds),
             vec3f(cellID)*rcp(vec3f(dims)));
    bounds.upper
      = lerp(getBox(mesh.worldBounds),
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

    ray.tMax = optixGetRayTmax();

    float delta = .1f;
    vec3f P[7];
    P[6] = ray.org + ray.tMax * ray.dir;
    P[0] = P[6] - delta * vec3f(1.f,0.f,0.f);
    P[1] = P[6] + delta * vec3f(1.f,0.f,0.f);
    P[2] = P[6] - delta * vec3f(0.f,1.f,0.f);
    P[3] = P[6] + delta * vec3f(0.f,1.f,0.f);
    P[4] = P[6] - delta * vec3f(0.f,0.f,1.f);
    P[5] = P[6] + delta * vec3f(0.f,0.f,1.f);

    vec4f mapped[7];
#pragma unroll
    for (int i=0;i<7;i++) {
      mapped[i] = self.sampleAndMap(P[i]);
    }

#pragma unroll
    for (int i=0;i<6;i++) {
      if (mapped[i] == vec4f(0.f)) {
        mapped[i] = mapped[6];
        P[i] = P[6];
      }
    }
    vec3f N;
    N.x = safeDiv(mapped[1].w-mapped[0].w, P[1].x-P[0].x);
    N.y = safeDiv(mapped[3].w-mapped[2].w, P[3].y-P[2].y);
    N.z = safeDiv(mapped[5].w-mapped[4].w, P[5].z-P[4].z);
    if (N == vec3f(0.f)) {
      N = ray.dir;
    }
    N = normalize(N);

    ray.hadHit = true;
    ray.hit.baseColor
      = getPos(mapped[6])
      // * (.3f+.7f*fabsf(dot(normalize(ray.dir),N)))
      ;
    ray.hit.N = N;
    ray.hit.P = P[6];
  }

  OPTIX_INTERSECT_PROGRAM(UMesh_MC_CUBQL_Isec)()
  {
    int primID = optixGetPrimitiveIndex();
    
    auto &ray = owl::getPRD<Ray>();
    const auto &self = getProgramData<UMeshAccel_MC_CUBQL::DD>();
    const auto &mesh = self.sampler.mesh;
    vec3i dims = self.mcGrid.dims;
    vec3i cellID(primID % dims.x,
                 (primID / dims.x) % dims.y,
                 primID / (dims.x*dims.y));

    // compute world-space bounds of macro cells
    box3f bounds;
    bounds.lower
      = lerp(getBox(mesh.worldBounds),
             vec3f(cellID)*rcp(vec3f(dims)));
    bounds.upper
      = lerp(getBox(mesh.worldBounds),
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

    // ray.hit.baseColor = vec3f(sample);
    ray.tMax = tRange.upper;
    optixReportIntersection(tRange.upper, 0);
  }
}
