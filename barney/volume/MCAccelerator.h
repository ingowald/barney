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
#ifdef __CUDA_ARCH__
#  include "owl/owl_device.h"
#  include "barney/volume/DDA.h"
#endif

namespace barney {
  using render::Ray;
  
  /*! a volume accel that creates an OWL geometry in the barney render
    graph; this is the base class that just defines the very concept
    of having a OWL geom, variables, etc; the actual sampler, accel
    struct, and traverser have to be added in derived classes */
  template<typename ScalarFieldType>
  struct OWLVolumeAccel {
    // /*! device-side data for this geom (this is what goes into the
    //   SBT); derived classes may add additional fields */
    // struct DD// : public VolumeAccel::DD<ScalarFieldType>
    // {
    //   VolumeDD<ScalarFieldType> volume;
    //   /*! declares this class' optix/owl device variables */
    //   // static void addVars(std::vector<OWLVarDecl> &vars, int base);
    // };
    /*! host-side code that implements the actual VolumeAccel */
    struct Host : public VolumeAccel {
      using Inherited = VolumeAccel;
      
      /* constuctor of host-side data */
      Host(ScalarField *sf, Volume *volume);

      UpdateMode updateMode() override
      { return HAS_ITS_OWN_GROUP; }

      // void writeDD(DD &dd, rtc::Device *device);
      // DD getDD(rtc::Device *device) { DD dd; return dd; }
        
      /*! set owl variables for this accelerator - this is virutal so
        derived classes can add their own */
      // void setVariables(OWLGeom geom) override;
      
      void build(bool full_rebuild) override;
      
      /*! creates the actual OWL geometry object that contains the
        prims that realize this volume accel. */
      virtual void createGeom() = 0;

      // /*! returns (part of) a string that allows to identify the
      //     device-side optix intersect/ch/bounds/etc functions that
      //     realize this geometry. Eg, if this accel uses macrocells,
      //     and all the device-functions are named after a scheme
      //     "<MyScalarType>_MC_<Bounds/Isec/CH/...>()" then this
      //     function should ask the scalar field type to create
      //     "<MyScalarType>", and append "_MC" to it */
      virtual std::string getTypeString() const
      { return sampler.getTypeString(); }

      typename ScalarFieldType::Host sampler;
      rtc::Geom *geom  = 0;
      rtc::Group *group = 0; 
      // const char  *const ptxCode;
    };
  };
  
  template<typename ScalarFieldType>
  struct MCVolumeAccel : public OWLVolumeAccel<ScalarFieldType>
  {
    struct DD // : public OWLVolumeAccel<ScalarFieldType>::DD
    {
      /*! declares this class' optix/owl device variables */
      // static void addVars(std::vector<OWLVarDecl> &vars, int base);
      
      Volume::DD<ScalarFieldType> volume;
      MCGrid::DD mcGrid;
    };
    
    struct Host : public OWLVolumeAccel<ScalarFieldType>::Host
    {
      using Inherited = typename OWLVolumeAccel<ScalarFieldType>::Host;
      
      using Inherited::sf;
      using Inherited::sampler;
      using Inherited::getXF;
      // using Inherited::ptxCode;
      using Inherited::geom;
      using Inherited::getTypeString;
      
      Host(ScalarField *sf, Volume *volume);
      // void setVariables(OWLGeom geom) override;
      void createGeom() override;

      void build(bool full_rebuild) override
      {
        assert(sf);
        if (mcGrid.dims.x == 0) {
          // macro cell grid hasn't even built ranges, yet -> let it do that.
          sf->buildMCs(mcGrid);
        }
        // update majorants on latest transfer function
        mcGrid.computeMajorants(getXF());
        Inherited::build(full_rebuild);
      };
      
      MCGrid       mcGrid;
    };
  };
    
  /*! a OWL geometry that creates one optix geom for each macrocell,
    uses refitting to 'hide' in-active ones, and leaves traversal of
    these macrocells to optix's traversal of the individual per-mc
    prims */
  template<typename ScalarFieldType>
  struct MCRTXVolumeAccel : public MCVolumeAccel<ScalarFieldType> {
    struct DD : public MCVolumeAccel<ScalarFieldType>::DD {};

#ifdef __CUDA_ARCH__
    /*! optix bounds prog for this class of accels */
    static inline __device__
    void boundsProg(const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __device__ void isProg();
    /*! optix closest-hit prog for this class of accels */
    static inline __device__ void chProg();
#endif

    struct Host : public MCVolumeAccel<ScalarFieldType>::Host {
      using Inherited = typename MCVolumeAccel<ScalarFieldType>::Host;
      
      using Inherited::sampler;
      using Inherited::geom;
      using Inherited::mcGrid;
      
      Host(ScalarField *sf, Volume *volume)
        : Inherited(sf,volume)
      {}
      
      /*! builds the string that allows for properly matching optix
        device progs for this type */
      std::string getTypeString() const override
      { return sampler.getTypeString()+"_MCRTX"; }
      
      void createGeom() override
      {
        // let parent class create the geometry itself, _we_ then only
        // set the numprims
        Inherited::createGeom();

        // this is a *RTXTraverser* for the mcgrid, we have one prim
        // per cell:
        const int primCount = mcGrid.dims.x*mcGrid.dims.y*mcGrid.dims.z;
        // owlGeomSetPrimCount(geom,primCount);
        geom->setPrimCount(primCount);
      }
    };
  };

  /*! a OWL geometry that creates a *SINGLE* optix primitive for the
    entire macro cell grid, then performs DDA traversal within that
    grid */
  template<typename ScalarFieldType>
  struct MCDDAVolumeAccel : public MCVolumeAccel<ScalarFieldType> {
    using DD = typename MCVolumeAccel<ScalarFieldType>::DD;
    
#ifdef __CUDA_ARCH__
    /*! optix bounds prog for this class of accels */
    static inline __device__
    void boundsProg(const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __device__ void isProg();
    /*! optix closest-hit prog for this class of accels */
    static inline __device__ void chProg();
#endif

    struct Host : public MCVolumeAccel<ScalarFieldType>::Host {
      using Inherited = typename MCVolumeAccel<ScalarFieldType>::Host;
      using Inherited::sampler;
      using Inherited::geom;
      
      Host(ScalarField *sf, Volume *volume)
        : Inherited(sf,volume)
      {}
      
      /*! builds the string that allows for properly matching optix
        device progs for this type */
      std::string getTypeString() const override
      { return sampler.getTypeString()+"_MCDDA"; }
      
      void createGeom() override
      {
        // let parent class create the geometry itself, _we_ then only
        // set the numprims
        Inherited::createGeom();
        
        // this is a *DDATraverser* for the mcgrid, we have one prim
        // over the whole grid
        const int primCount = 1;
        geom->setPrimCount(primCount);
      }
    };
  };
  
  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================


  /* constuctor of host-side data */
  template<typename ScalarFieldType>
  OWLVolumeAccel<ScalarFieldType>::Host::Host(ScalarField *sf, Volume *volume)
    : VolumeAccel(sf,volume),
      sampler(sf)
  {}

  /*! set owl variables for this accelerator - this is virutal so
    derived classes can add their own */
  // template<typename ScalarFieldType>
  // void OWLVolumeAccel<ScalarFieldType>::Host::setVariables(OWLGeom geom) 
  // {
  //   Inherited::setVariables(geom);
  //   sf->setVariables(geom);
  // }

  // template<typename ScalarFieldType>
  // void OWLVolumeAccel<ScalarFieldType>::Host
  // ::writeDD(typename OWLVolumeAccel<ScalarFieldType>::DD &dd,
  //           rtc::Device *device) 
  // {
  //   Inherited::writeDD(dd,device);
  //   sf->writeDD(dd,device);
  //   // Inherited::setVariables(geom);
  //   // sf->setVariables(geom);
  // }
  
  template<typename ScalarFieldType>
  void OWLVolumeAccel<ScalarFieldType>::Host::build(bool full_rebuild) 
  {
    if (!geom) {
      /*! first time build needs to create the actual OWL geom object;
        this is done in virtual function because only derived
        classes will know full geometry type */
      createGeom();
      // group = owlUserGeomGroupCreate(this->getOWL(),1,&geom);
      group = devGroup->rtc->createUserGeomsGroup({geom});
      volume->generatedGroups.clear();
      volume->generatedGroups.push_back(group);
    }
    sampler.build(full_rebuild);

    for (auto device : devGroup->devices) {
      typename OWLVolumeAccel<ScalarFieldType>::DD dd;
#if 1
      BARNEY_NYI();
#else
      dd = getDD(device->rtc);
#endif
      // writeDD(dd,device->rtc);
      geom->setDD(&dd,device->rtc);
    }
    // setVariables(geom);
    // owlGroupBuildAccel(group);
    group->buildAccel();
  };
  

  // /*! declares this class' optix/owl device variables */
  // template<typename ScalarFieldType>
  // void OWLVolumeAccel<ScalarFieldType>::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  // {
  //   VolumeAccel::DD<ScalarFieldType>::addVars(vars,base);
  // }



  // /*! declares this class' optix/owl device variables */
  // template<typename ScalarFieldType>
  // void MCVolumeAccel<ScalarFieldType>::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  // {
  //   using Inherited = typename OWLVolumeAccel<ScalarFieldType>::DD;
  //   Inherited::addVars(vars,base);
  //   MCGrid::DD::addVars(vars,base+OWL_OFFSETOF(DD,mcGrid));
  // }



  template<typename ScalarFieldType>
  MCVolumeAccel<ScalarFieldType>::Host::Host(ScalarField *sf,
                                       Volume *volume)
    : Inherited(sf,volume),
      mcGrid(sf->getDevGroup())
  {}

  // template<typename ScalarFieldType>
  // void MCVolumeAccel<ScalarFieldType>::Host::setVariables(OWLGeom geom) 
  // {
  //   Inherited::setVariables(geom);
  //   mcGrid.setVariables(geom);
  //   sampler.setVariables(geom);
  // }

  template<typename ScalarFieldType>
  void MCVolumeAccel<ScalarFieldType>::Host::createGeom() 
  {
    auto devGroup = sf->getDevGroup();
    const std::string typeString
      = getTypeString();
    rtc::GeomType *gt = devGroup->getOrCreateGeomTypeFor
      (typeString, [&](DevGroup *dg)
      {
        assert(dg);
        // std::vector<OWLVarDecl> params;
        // MCVolumeAccel<ScalarFieldType>::DD::addVars(params,0);
        // OWLGeomType gt = owlGeomTypeCreate
        //   (dg->owl,OWL_GEOM_USER,
        //    sizeof(MCVolumeAccel<ScalarFieldType>::DD),params.data(),(int)params.size());
        // OWLModule module = owlModuleCreate(dg->owl,ptxCode);
        const std::string boundsProg = typeString+"_Bounds";
        const std::string isProg = typeString+"_Isec";
        const std::string chProg = typeString+"_CH";
        // owlGeomTypeSetBoundsProg(gt,module,boundsProg.c_str());
        // owlGeomTypeSetIntersectProg(gt,0,module,isProg.c_str());
        // owlGeomTypeSetClosestHit(gt,0,module,chProg.c_str());
        // owlBuildPrograms(dg->owl);
        // return gt;
        return devGroup->rtc->createUserGeomType(typeString.c_str(),
                                                 sizeof(DD),
                                                 boundsProg.c_str(),
                                                 isProg.c_str(),
                                                 nullptr,
                                                 chProg.c_str());
      });
        
    // geom = owlGeomCreate(devGroup->owl,gt);
    geom = devGroup->rtc->createGeom(gt);
    // geom = owlGeomCreate(devGroup->owl,gt);
    // this CREATES the geom, but doesn't yet set prim count (prim
    // count depends on how the mc grid is traersed), so has to be
    // set in the derived function
  }
      
  






  // ------------------------------------------------------------------
  // device progs: macro-cell accel with RTX traversal
  // ------------------------------------------------------------------

#ifdef __CUDA_ARCH__
  template<typename ScalarFieldType>
  inline __device__
  void MCRTXVolumeAccel<ScalarFieldType>::boundsProg(const void *geomData,
                                               owl::common::box3f &bounds,
                                               const int32_t primID)
  {
    // "RTX": we need to create one prim per macro cell
    
    // for now, do not use refitting, simple do rebuild every
    // frame... in this case we can simply return empty box for every
    // inactive cell.

    const DD &self = *(DD*)geomData;
    if (primID >= self.mcGrid.numCells()) return;
    
    const float maj = self.mcGrid.majorants[primID];
    if (maj == 0.f) {
      bounds = box3f();
    } else {
      const vec3i mcID = self.mcGrid.cellID(primID);
      bounds = self.mcGrid.cellBounds(mcID,self.volume.sf.worldBounds);
    }
  }

  template<typename ScalarFieldType>
  inline __device__
  void MCRTXVolumeAccel<ScalarFieldType>::isProg()
  {
    /* ALL of this code should be exactly the same in any
       instantiation of the MCRTXVolumeAccel<> tempalte! */
    const DD &self = owl::getProgramData<DD>();
    Ray &ray = owl::getPRD<Ray>();

    
    vec3f org = optixGetObjectRayOrigin();
    vec3f dir = optixGetObjectRayDirection();
    
    const int primID = optixGetPrimitiveIndex();

    const vec3i mcID = self.mcGrid.cellID(primID);
    
    const float majorant = self.mcGrid.majorants[primID];
    if (majorant == 0.f) return;
    
    box3f bounds = self.mcGrid.cellBounds(mcID,self.volume.sf.worldBounds);
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };

    if (!boxTest(ray,tRange,bounds))
      return;

    vec4f sample = 0.f;
    if (!Woodcock::sampleRange(sample,self.volume,
                               org,dir,tRange,majorant,ray.rngSeed,
                               ray.dbg))
      return;

    // and: store the hit, right here in isec prog.
    ray.setVolumeHit(ray.org + tRange.upper*ray.dir,
                     tRange.upper,
                     getPos(sample));
    optixReportIntersection(tRange.upper, 0);
  }

  template<typename ScalarFieldType>
  inline __device__
  void MCRTXVolumeAccel<ScalarFieldType>::chProg()
  {/* nothing - already all set in isec */}
#endif


  // ------------------------------------------------------------------
  // device progs: macro-cell accel with DDA traversal
  // ------------------------------------------------------------------


#ifdef __CUDA_ARCH__
  template<typename ScalarFieldType>
  inline __device__
  void MCDDAVolumeAccel<ScalarFieldType>::boundsProg(const void *geomData,
                                               owl::common::box3f &bounds,
                                               const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.volume.sf.worldBounds;
  }

  template<typename ScalarFieldType>
  inline __device__
  void MCDDAVolumeAccel<ScalarFieldType>::isProg()
  {
    /* ALL of this code should be exactly the same in any
       instantiation of the MCDDAVolumeAccel<> tempalte! */
    const DD &self = owl::getProgramData<DD>();
    Ray &ray = owl::getPRD<Ray>();

    box3f bounds = self.volume.sf.worldBounds;
    range1f tRange = { optixGetRayTmin(), optixGetRayTmax() };

    if (!boxTest(ray,tRange,bounds))
      return;
    
    // ray in world space
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
                optixReportIntersection(tRange.upper, 0);
                return false;
              },
              /*NO debug*/false
              );
  }
    
  template<typename ScalarFieldType>
  inline __device__
  void MCDDAVolumeAccel<ScalarFieldType>::chProg()
  {/* nothing - already all set in isec */}
#endif

}
