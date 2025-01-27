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
  
  /*! a volume accel that creates an OWL geometry in the barney render
    graph; this is the base class that just defines the very concept
    of having a OWL geom, variables, etc; the actual sampler, accel
    struct, and traverser have to be added in derived classes */
  // template<typename SFType>
  // struct OWLVolumeAccel {
  //   // /*! device-side data for this geom (this is what goes into the
  //   //   SBT); derived classes may add additional fields */
  //   // struct DD// : public VolumeAccel::DD<SFType>
  //   // {
  //   //   VolumeDD<SFType> volume;
  //   //   /*! declares this class' optix/owl device variables */
  //   //   // static void addVars(std::vector<OWLVarDecl> &vars, int base);
  //   // };
  //   /*! host-side code that implements the actual VolumeAccel */
  //   struct Host : public VolumeAccel {
  //     using Inherited = VolumeAccel;
      
  //     /* constuctor of host-side data */
  //     Host(ScalarField *sf, Volume *volume);

  //     UpdateMode updateMode() override
  //     { return HAS_ITS_OWN_GROUP; }

  //     // void writeDD(DD &dd, rtc::Device *device);
  //     // DD getDD(rtc::Device *device) { DD dd; return dd; }
        
  //     /*! set owl variables for this accelerator - this is virutal so
  //       derived classes can add their own */
  //     // void setVariables(OWLGeom geom) override;
      
  //     void build(bool full_rebuild) override;
      
  //     /*! creates the actual OWL geometry object that contains the
  //       prims that realize this volume accel. */
  //     virtual rtc::Geom *createGeom(Device *device) = 0;

  //     // /*! returns (part of) a string that allows to identify the
  //     //     device-side optix intersect/ch/bounds/etc functions that
  //     //     realize this geometry. Eg, if this accel uses macrocells,
  //     //     and all the device-functions are named after a scheme
  //     //     "<MyScalarType>_MC_<Bounds/Isec/CH/...>()" then this
  //     //     function should ask the scalar field type to create
  //     //     "<MyScalarType>", and append "_MC" to it */
  //     virtual std::string getTypeString() const = 0;

  //     // typename SFType::Host sampler;
      
  //     struct PLD {
  //       rtc::Geom *geom  = 0;
  //       rtc::Group *group = 0; 
  //     };
  //     PLD *getPLD(Device *device) 
  //     { return &perLogical[device->contextRank]; } 
  //     std::vector<PLD> perLogical;
  //   };
  // };
  
  template<typename SFType>
  struct MCVolumeAccel : public VolumeAccel //OWLVolumeAccel<SFType>
  {
    struct DD // : public OWLVolumeAccel<SFType>::DD
    {
      /*! declares this class' optix/owl device variables */
      // static void addVars(std::vector<OWLVarDecl> &vars, int base);
      int beginSentinel = 0x1234;
      Volume::DD<SFType> volume;
      MCGrid::DD mcGrid;
      int endSentinel = 0x5678;
    };

    static rtc::GeomType *createGeomType(rtc::Device *device);
    
    static std::string getTypeName()
    { return "MCAccel_"+SFType::typeName(); }
    
    struct PLD {
      rtc::Geom  *geom  = 0;
      rtc::Group *group  = 0;
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
    std::cout << OWL_TERMINAL_GREEN
              << "creating '" << typeName << "' Volume type"
              << OWL_TERMINAL_DEFAULT << std::endl;

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
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        std::string typeName = getTypeName();
        rtc::GeomType *gt
          = device->geomTypes.get(typeName,
                                  MCVolumeAccel<SFType>::createGeomType);
        rtc::Geom *geom = gt->createGeom();
        pld->geom  = geom;
        geom->setPrimCount(1);
        pld->group = device->rtc->createUserGeomsGroup({geom});
      }
      rtc::Group *group = pld->group;

      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty()) {
        volumePLD->generatedGroups = {group};
      }

      DD dd = getDD(device);
      pld->geom->setDD(&dd);
      group->buildAccel();
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
