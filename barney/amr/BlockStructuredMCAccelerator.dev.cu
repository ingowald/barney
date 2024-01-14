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

#include "barney/amr/BlockStructuredCUBQLSampler.h"
// #include "barney/volume/DDA.h"
// #include <owl/owl_device.h>
// #include "barney/amr/BlockStructuredCU.h"
// // #include "barney/amr/BlockStructuredMCAccelerator.h"
// #include <owl/owl_device.h>

namespace barney {


  OPTIX_BOUNDS_PROGRAM(BlockStructured_MCRTX_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    // #if 1
    MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg
      (geomData,bounds,primID);
    // #else
    //     // "RTX": we need to create one prim per macro cell
    
    //     // for now, do not use refitting, simple do rebuild every
    //     // frame... in this case we can simply return empty box for every
    //     // inactive cell.

    //     using Geom = typename MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::DD;
    //     const Geom &self = *(Geom*)geomData;
    //     if (primID >= self.mcGrid.numCells()) return;
    
    //     const float maj = self.mcGrid.majorants[primID];
    //     if (maj == 0.f) {
    //       bounds = box3f();
    //     } else {
    //       const vec3i mcID = self.mcGrid.cellID(primID);
    //       bounds = self.mcGrid.cellBounds(mcID,self.worldBounds);
    //     }
    // #endif
  }

  OPTIX_INTERSECT_PROGRAM(BlockStructured_MCRTX_Isec)()
  {
    // #if 1
    MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::isProg();
    // #else
    //     /* ALL of this code should be exactly the same in any
    //        instantiation of the MCRTXVolumeAccel<> tempalte! */
    //     using Geom = typename MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::DD;
    //     const Geom &self = owl::getProgramData<Geom>();
    //     Ray &ray = owl::getPRD<Ray>();
    //     vec3f org = optixGetObjectRayOrigin();
    //     vec3f dir = optixGetObjectRayDirection();
    //     const int primID = optixGetPrimitiveIndex();

    //     const vec3i mcID = self.mcGrid.cellID(primID);
    
    //     const float majorant = self.mcGrid.majorants[primID];
    //     if (majorant == 0.f) return;
    
    //     box3f bounds = self.mcGrid.cellBounds(mcID,self.worldBounds);
    //     range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };

    //     if (!boxTest(ray,tRange,bounds))
    //       return;
    
    //     vec4f sample = 0.f;
    //     if (!Woodcock::sampleRange(sample,self,
    //                                org,dir,tRange,majorant,ray.rngSeed
    //                                //,ray.dbg
    //                                ))
    //       return;

    //     // and: store the hit, right here in isec prog.
    //     ray.tMax          = tRange.upper;
    //     ray.hit.baseColor = getPos(sample);
    //     ray.hit.N         = vec3f(0.f);
    //     ray.hit.P         = ray.org + tRange.upper*ray.dir;
    //     optixReportIntersection(tRange.upper, 0);
    // #endif
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MCRTX_CH)()
  {
    /* nothing - already all set in isec */
    MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::chProg();
  }
  





  OPTIX_BOUNDS_PROGRAM(BlockStructured_MCDDA_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    // #if 1
    MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::boundsProg(geomData,bounds,primID);
    // #else
    //     // "DDA": we have a single prim for entire volume, iteration over
    //     // cells happens in DDA-traversal in IS prog.

    //     using Geom = typename MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::DD;
    //     const Geom &self = *(Geom*)geomData;
    //     bounds = self.worldBounds;
    // #endif
  }

  OPTIX_INTERSECT_PROGRAM(BlockStructured_MCDDA_Isec)()
  {
    // #if 1
    MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::isProg();
    // #else
    //     /* ALL of this code should be exactly the same in any
    //        instantiation of the MCDDAVolumeAccel<> tempalte! */
    //     using Geom = typename MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::DD;
    //     const Geom &self = owl::getProgramData<Geom>();
    //     Ray &ray = owl::getPRD<Ray>();

    //     box3f bounds = self.worldBounds;
    //     range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };
    
    //     if (!boxTest(ray,tRange,bounds))
    //       return;
    
    //     vec3f obj_org = optixGetObjectRayOrigin();
    //     vec3f obj_dir = optixGetObjectRayDirection();

    //     // ------------------------------------------------------------------
    //     // compute ray in macro cell grid space 
    //     // ------------------------------------------------------------------
    //     vec3f mcGridOrigin = self.mcGrid.gridOrigin;
    //     vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    //     vec3f dda_org = obj_org;
    //     vec3f dda_dir = obj_dir;
    
    //     dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    //     dda_dir = dda_dir * rcp(mcGridSpacing);
    
    //     dda::dda3(dda_org,dda_dir,tRange.upper,
    //               vec3ui(self.mcGrid.dims),
    //               [&](const vec3i &cellIdx, float t0, float t1) -> bool
    //               {
    //                 float majorant = self.mcGrid.majorant(cellIdx);
    //                 if (majorant == 0.f) return true;
                
    //                 vec4f sample = 0.f;
    //                 range1f tRange = {t0,t1};
    //                 if (!Woodcock::sampleRange(sample,self,
    //                                            obj_org,obj_dir,
    //                                            tRange,majorant,ray.rngSeed))
    //                   return true;
                
    //                 ray.tMax          = tRange.upper;
    //                 ray.hit.baseColor = getPos(sample);
    //                 ray.hit.N         = vec3f(0.f);
    //                 ray.hit.P         = ray.org + tRange.upper*ray.dir;
    //                 optixReportIntersection(tRange.upper, 0);
    //                 return false;
    //               },
    //               /*NO debug*/false);
    // #endif
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(BlockStructured_MCDDA_CH)()
  {
    MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::chProg();
    /* nothing - already all set in isec */
  }


  
//   OPTIX_BOUNDS_PROGRAM(BlockBlockStructured_MC_CUBQL_Bounds)(const void *geomData,
//                                                         owl::common::box3f &bounds,
//                                                         const int32_t primID)
//   {
//     auto self = *(const BlockBlockStructuredAccel_MC_CUBQL::DD*)geomData;
//     vec3i dims = self.mcGrid.dims;
//     const auto &field = self.sampler.field;
//     if (primID >= dims.x*dims.y*dims.z)
//       return;

//     vec3i cellID(primID % dims.x,
//                  (primID / dims.x) % dims.y,
//                  primID / (dims.x*dims.y));

//     // compute world-space bounds of macro cells
//     bounds.lower
//       = lerp(getBox(field.worldBounds),
//              vec3f(cellID)*rcp(vec3f(dims)));
//     bounds.upper
//       = lerp(getBox(field.worldBounds),
//              vec3f(cellID+vec3i(1))*rcp(vec3f(dims)));

//     // printf("bounds (%f %f %f)(%f %f %f)\n",
//     //        bounds.lower.x,
//     //        bounds.lower.y,
//     //        bounds.lower.z,
//     //        bounds.upper.x,
//     //        bounds.upper.y,
//     //        bounds.upper.z);
    
// # if 1
//     // kill this prim is majorant is 0
//     if (self.mcGrid.majorants &&
//         self.mcGrid.majorants[primID] == 0.f)
//       bounds = box3f();
// # endif
//   }

//   OPTIX_CLOSEST_HIT_PROGRAM(BlockBlockStructured_MC_CUBQL_CH)()
//   {
//     auto &ray = owl::getPRD<Ray>();
//     const auto &self = getProgramData<BlockBlockStructuredAccel_MC_CUBQL::DD>();
//     int primID = optixGetPrimitiveIndex();

//     ray.tMax = optixGetRayTmax();

//     vec3f P = ray.org + ray.tMax * ray.dir;
//     vec4f mapped = self.sampleAndMap(P);

//     ray.hadHit = true;
//     ray.hit.baseColor
//       = getPos(mapped)
//       // * (.3f+.7f*fabsf(dot(normalize(ray.dir),N)))
//       ;
//     ray.hit.N = vec3f(0.f);
//     ray.hit.P = P[6];
//   }

//   OPTIX_INTERSECT_PROGRAM(BlockBlockStructured_MC_CUBQL_Isec)()
//   {
//     int primID = optixGetPrimitiveIndex();
    
//     auto &ray = owl::getPRD<Ray>();
//     const auto &self = getProgramData<BlockBlockStructuredAccel_MC_CUBQL::DD>();
//     const auto &field = self.sampler.field;
//     vec3i dims = self.mcGrid.dims;
//     vec3i cellID(primID % dims.x,
//                  (primID / dims.x) % dims.y,
//                  primID / (dims.x*dims.y));
    
//     // compute world-space bounds of macro cells
//     box3f bounds;
//     bounds.lower
//       = lerp(getBox(field.worldBounds),
//              vec3f(cellID)*rcp(vec3f(dims)));
//     bounds.upper
//       = lerp(getBox(field.worldBounds),
//              vec3f(cellID+vec3i(1))*rcp(vec3f(dims)));
//     range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };
    
//     if (!boxTest(ray,tRange,bounds))
//       return;

//     float majorant = self.mcGrid.majorants[primID];
//     vec4f sample = 0.f;
//     if (!Woodcock::sampleRange(sample,self,
//                                ray.org,ray.dir,tRange,majorant,ray.rngSeed,
//                                ray.dbg))
//       return;

//     ray.tMax = tRange.upper;
//     optixReportIntersection(tRange.upper, 0);
//   }
}
