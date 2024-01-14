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

/*! \file UMeshCUBQLMC.dev.cu implements the DDA and RTX traversers for a umesh
    scalar field with cubql sampler and macro cell accel */

#include "barney/umesh/UMeshCUBQLSampler.h"
#include "barney/volume/DDA.h"
#include <owl/owl_device.h>

namespace barney {

  OPTIX_BOUNDS_PROGRAM(UMesh_CUBQL_MCRTX_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    // "RTX": we need to create one prim per macro cell
    
    // for now, do not use refitting, simple do rebuild every
    // frame... in this case we can simply return empty box for every
    // inactive cell.

    using Geom = typename MCRTXVolumeAccel<UMeshCUBQLSampler>::DD;
    const Geom &self = *(Geom*)geomData;
    if (primID >= self.mcGrid.numCells()) return;
    
    const float maj = self.mcGrid.majorants[primID];
    if (maj == 0.f) {
      bounds = box3f();
    } else {
      const vec3i mcID = self.mcGrid.cellID(primID);
      bounds = self.mcGrid.cellBounds(mcID,self.worldBounds);
    }
  }

  OPTIX_INTERSECT_PROGRAM(UMesh_CUBQL_MCRTX_Isec)()
  {
    /* ALL of this code should be exactly the same in any
       instantiation of the MCRTXVolumeAccel<> tempalte! */
    using Geom = typename MCRTXVolumeAccel<UMeshCUBQLSampler>::DD;
    const Geom &self = owl::getProgramData<Geom>();
    Ray &ray = owl::getPRD<Ray>();
    vec3f org = optixGetObjectRayOrigin();
    vec3f dir = optixGetObjectRayDirection();
    const int primID = optixGetPrimitiveIndex();

    const vec3i mcID = self.mcGrid.cellID(primID);
    
    const float majorant = self.mcGrid.majorants[primID];
    if (majorant == 0.f) return;
    
    box3f bounds = self.mcGrid.cellBounds(mcID,self.worldBounds);
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };

    if (!boxTest(ray,tRange,bounds))
      return;
    
    vec4f sample = 0.f;
    if (!Woodcock::sampleRange(sample,self,
                               org,dir,tRange,majorant,ray.rngSeed
                               //,ray.dbg
                               ))
      return;

    // and: store the hit, right here in isec prog.
    ray.tMax          = tRange.upper;
    ray.hit.baseColor = getPos(sample);
    ray.hit.N         = vec3f(0.f);
    ray.hit.P         = ray.org + tRange.upper*ray.dir;
    optixReportIntersection(tRange.upper, 0);
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(UMesh_CUBQL_MCRTX_CH)()
  {
    /* nothing - already all set in isec */
  }
  





  OPTIX_BOUNDS_PROGRAM(UMesh_CUBQL_MCDDA_Bounds)(const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    // "DDA": we have a single prim for entire volume, iteration over
    // cells happens in DDA-traversal in IS prog.

    using Geom = typename MCDDAVolumeAccel<UMeshCUBQLSampler>::DD;
    const Geom &self = *(Geom*)geomData;
    bounds = self.worldBounds;
  }

  OPTIX_INTERSECT_PROGRAM(UMesh_CUBQL_MCDDA_Isec)()
  {
    /* ALL of this code should be exactly the same in any
       instantiation of the MCDDAVolumeAccel<> tempalte! */
    using Geom = typename MCDDAVolumeAccel<UMeshCUBQLSampler>::DD;
    const Geom &self = owl::getProgramData<Geom>();
    Ray &ray = owl::getPRD<Ray>();

    box3f bounds = self.worldBounds;
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };
    
    if (!boxTest(ray,tRange,bounds))
      return;
    
    vec3f obj_org = optixGetObjectRayOrigin();
    vec3f obj_dir = optixGetObjectRayDirection();

    // ------------------------------------------------------------------
    // compute ray in macro cell grid space 
    // ------------------------------------------------------------------
    vec3f mcGridOrigin = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    vec3f dda_org = obj_org;
    vec3f dda_dir = obj_dir;
    
    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

    bool dbg = ray.dbg;
    if (dbg) printf("CALLING DDA\n");
    dda::dda3(dda_org,dda_dir,tRange.upper,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
                float majorant = self.mcGrid.majorant(cellIdx);
                if (majorant == 0.f) return true;
                
                vec4f sample = 0.f;
                range1f tRange = {t0,t1};
                if (!Woodcock::sampleRange(sample,self,
                                           obj_org,obj_dir,
                                           tRange,majorant,ray.rngSeed))
                  return true;
                
                ray.tMax          = tRange.upper;
                ray.hit.baseColor = getPos(sample);
                ray.hit.N         = vec3f(0.f);
                ray.hit.P         = ray.org + tRange.upper*ray.dir;
                optixReportIntersection(tRange.upper, 0);
                return false;
              },
              /*NO debug*/dbg);
  }
  
  OPTIX_CLOSEST_HIT_PROGRAM(UMesh_CUBQL_MCDDA_CH)()
  {
    /* nothing - already all set in isec */
  }
  
}

