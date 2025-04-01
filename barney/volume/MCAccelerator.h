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

#pragma once

#include "barney/DeviceGroup.h"
#include "barney/volume/Volume.h"
#include "barney/volume/MCGrid.h"
#include "barney/volume/DDA.h"

namespace BARNEY_NS {
  using render::Ray;
 
  template<typename SFSampler>
  struct MCVolumeAccel : public VolumeAccel 
  {
    struct DD 
    {
      Volume::DD<SFSampler> volume;
      MCGrid::DD            mcGrid;
    };

    struct PLD {
      rtc::Geom  *geom  = 0;
      rtc::Group *group = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank]; } 
    std::vector<PLD> perLogical;

    DD getDD(Device *device)
    {
      DD dd;
      dd.volume = volume->getDD(device,sfSampler);
      dd.mcGrid = mcGrid.getDD(device);
      return dd;
    }

    MCVolumeAccel(Volume *volume,
                  GeomTypeCreationFct creatorFct,
                  const std::shared_ptr<SFSampler> &sfSampler);
      // ,
      //             const std::string &embeddedPTXStringName,
      //             const std::string &programsTypeName);

      GeomTypeCreationFct const creatorFct;
    
    void build(bool full_rebuild) override;
    
#if BARNEY_DEVICE_PROGRAM
    /*! optix bounds prog for this class of accels */
    static inline __rtc_device
    void boundsProg(const rtc::TraceInterface &ti,
                    const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __rtc_device
    void isProg(rtc::TraceInterface &ti);
    /*! optix closest-hit prog for this class of accels */
#endif
    
    MCGrid       mcGrid;
    const std::shared_ptr<SFSampler> sfSampler;
  };
  
  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================

  template<typename SFSampler>
  void MCVolumeAccel<SFSampler>::build(bool full_rebuild) 
  {
    sfSampler->build();
    if (mcGrid.dims.x == 0)
      volume->sf->buildMCs(mcGrid);
    mcGrid.computeMajorants(&volume->xf);
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFct);
        // build a single-prim geometry, that single prim is our
        // entire MC/DDA grid
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
      }
      rtc::Geom *geom = pld->geom;
      DD dd = getDD(device);
      geom->setDD(&dd);

      if (!pld->group) {
        // now put that into a instantiable group, and build it.
        pld->group = device->rtc->createUserGeomsGroup({geom});
      }
      pld->group->buildAccel();

      // now let the actual volume we're building know about the
      // group we just created
      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty()) 
        volumePLD->generatedGroups = { pld->group };
    }
  }
  

  template<typename SFSampler>
  MCVolumeAccel<SFSampler>::
  MCVolumeAccel(Volume *volume,
                GeomTypeCreationFct creatorFct,
                const std::shared_ptr<SFSampler> &sfSampler)
    : VolumeAccel(volume),
      mcGrid(volume->sf->devices),
      sfSampler(sfSampler),
      creatorFct(creatorFct)
  {
    perLogical.resize(devices->numLogical);
  }

  // ------------------------------------------------------------------
  // device progs: macro-cell accel with DDA traversal
  // ------------------------------------------------------------------

#if BARNEY_DEVICE_PROGRAM
  template<typename SFSampler>
  inline __rtc_device
  void MCVolumeAccel<SFSampler>::boundsProg(const rtc::TraceInterface &ti,
                                            const void *geomData,
                                            owl::common::box3f &bounds,
                                            const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.volume.sfCommon.worldBounds;
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCVolumeAccel<SFSampler>::isProg(rtc::TraceInterface &ti)
  {
    const void *pd = ti.getProgramData();
           
    const DD &self = *(typename MCVolumeAccel<SFSampler>::DD*)pd;
    // ray in world space
    Ray &ray = *(Ray*)ti.getPRD();
    
    box3f bounds = self.volume.sfCommon.worldBounds;
    range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
    
    // ray in object space
    vec3f obj_org = ti.getObjectRayOrigin();
    vec3f obj_dir = ti.getObjectRayDirection();

    auto objRay = ray;
    objRay.org = obj_org;
    objRay.dir = obj_dir;

    if (!boxTest(objRay,tRange,bounds))
      return;
    
    // ------------------------------------------------------------------
    // compute ray in macro cell grid space 
    // ------------------------------------------------------------------
    vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    // printf("grid %f %f %f / %f %f %f\n",
    //        mcGridOrigin.x,
    //        mcGridOrigin.y,
    //        mcGridOrigin.z,
    //        mcGridSpacing.x,
    //        mcGridSpacing.y,
    //        mcGridSpacing.z);
    vec3f dda_org = obj_org;
    vec3f dda_dir = obj_dir;

    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

    // printf("isec\n");
    dda::dda3(dda_org,dda_dir,tRange.upper,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
                // printf("dda %i %i %i \n",
                //        cellIdx.x,
                //        cellIdx.y,
                //        cellIdx.z);
                const float majorant = self.mcGrid.majorant(cellIdx);
                
                if (majorant == 0.f) return true;
                
                vec4f   sample = 0.f;
                range1f tRange = {t0,min(t1,ray.tMax)};
                if (!Woodcock::sampleRange(sample,self.volume,
                                           obj_org,obj_dir,
                                           tRange,majorant,ray.rngSeed,
                                           ray.dbg)) 
                  return true;
                
                vec3f P = ray.org + tRange.upper*ray.dir;
                ray.setVolumeHit(P,
                                 tRange.upper,
                                 getPos(sample));
                ti.reportIntersection(tRange.upper, 0);
                return false;
              },
              /*NO debug:*/false
              );
  }
#endif
}
