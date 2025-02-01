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

namespace barney {
  using render::Ray;
  
  template<typename SFType>
  struct MCVolumeAccel : public VolumeAccel 
  {
    struct DD 
    {
      Volume::DD<SFType> volume;
      MCGrid::DD mcGrid;
    };

    static rtc::GeomType *createGeomType(rtc::Device *device);
    
    static std::string getTypeName()
    { return "MCAccel_"+SFType::typeName(); }
    
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
      dd.mcGrid = mcGrid.getDD(device);
      dd.volume = volume->getDD<SFType>(device);
      return dd;
    }
    
    MCVolumeAccel(ScalarField *sf, Volume *volume);
    
    void build(bool full_rebuild) override;
    
    /*! optix bounds prog for this class of accels */
    template<typename TraceInterface>
    static inline __both__
    void boundsProg(TraceInterface &ti,
                    const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    template<typename TraceInterface>
    static inline __both__ void isProg(TraceInterface &ti);
    /*! optix closest-hit prog for this class of accels */
  
    static std::string getTypeString() 
    { return SFType::getTypeString(); }
    
    MCGrid       mcGrid;
  };
  
  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================


  template<typename SFType>
  rtc::GeomType *MCVolumeAccel<SFType>::createGeomType(rtc::Device *device)
  {
    std::string typeName = getTypeName();
    // std::cout << OWL_TERMINAL_GREEN
    //           << "creating '" << typeName << "' Volume type"
    //           << OWL_TERMINAL_DEFAULT << std::endl;

    return device->createUserGeomType(typeName.c_str(),
                                      sizeof(DD),
                                      /*ah*/false,/*ch*/false);
  }
  

  template<typename SFType>
  void MCVolumeAccel<SFType>::build(bool full_rebuild) 
  {
    if (mcGrid.dims.x == 0)
      sf->buildMCs(mcGrid);
    mcGrid.computeMajorants(&volume->xf);
    
    for (auto device : *devices) {
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        std::string typeName = getTypeName();
        rtc::GeomType *gt
          = device->geomTypes.get(typeName,
                                  MCVolumeAccel<SFType>::createGeomType);

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
  

  template<typename SFType>
  MCVolumeAccel<SFType>::MCVolumeAccel(ScalarField *sf,
                                                Volume *volume)
    : VolumeAccel(sf,volume),
      mcGrid(sf->devices)
  {
    perLogical.resize(devices->numLogical);
  }

  // ------------------------------------------------------------------
  // device progs: macro-cell accel with DDA traversal
  // ------------------------------------------------------------------


  template<typename SFType>
  template<typename TraceInterface>
  inline __both__
  void MCVolumeAccel<SFType>::boundsProg(TraceInterface &ti,
                                         const void *geomData,
                                         owl::common::box3f &bounds,
                                         const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.volume.sf.worldBounds;
  }
  
  template<typename SFType>
  template<typename TraceInterface>
  inline __both__
  void MCVolumeAccel<SFType>::isProg(TraceInterface &ti)
  {
    const void *pd = ti.getProgramData();
           
    const DD &self = *(typename MCVolumeAccel<SFType>::DD*)pd;
    Ray &ray = *(Ray*)ti.getPRD();
    
    box3f bounds = self.volume.sf.worldBounds;
    range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
    
    if (!boxTest(ray,tRange,bounds))
      return;
    
    // ray in world space
    vec3f obj_org = ti.getObjectRayOrigin();
    vec3f obj_dir = ti.getObjectRayDirection();

    // ------------------------------------------------------------------
    // compute ray in macro cell grid space 
    // ------------------------------------------------------------------
    vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    vec3f dda_org = obj_org;
    vec3f dda_dir = obj_dir;

    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

    // printf("dda grid dims %i %i %i\n",
    //        self.mcGrid.dims.x,
    //        self.mcGrid.dims.y,
    //        self.mcGrid.dims.z);
    dda::dda3(dda_org,dda_dir,tRange.upper,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
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
              ray.dbg//              /*NO debug*/false
              );
  }
  
  // template<typename SFType>
  // inline __both__
  // void MCVolumeAccel<SFType>::chProg()
  // {/* nothing - already all set in isec */}
  
}
