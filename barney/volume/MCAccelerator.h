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

namespace barney {

  template<typename SF>
  struct OWLVolumeAccel {
    struct DD : public SF::DD {
      static void addVars(std::vector<OWLVarDecl> &vars, int base)
      {
        SF::DD::addVars(vars,base);
      }
    };
    struct Host : public VolumeAccel {
      Host(ScalarField *sf, Volume *volume, const char *ptxCode)
        : VolumeAccel(sf,volume),
          ptxCode(ptxCode)
      {}

      UpdateMode updateMode() override
      { return HAS_ITS_OWN_GROUP; }

      virtual void setVariables(OWLGeom geom) 
      { sf->setVariables(geom); }
      
      void build(bool full_rebuild) override
      {
        PING; PRINT(full_rebuild);
        if (!geom) {
          createGeom();
          std::cout << "#bn.mc: geometry created" << std::endl;
          PRINT(geom);
          group = owlUserGeomGroupCreate(this->getOWL(),1,&geom);
          PING;
          volume->generatedGroups = { group }; 
          PRINT(volume->generatedGroups.size());
        }

        setVariables(geom);
        owlGroupBuildAccel(group);
      };

      /*! creates the actual OWL geometry object that contains the
          prims that realize this volume accel. */
      virtual void createGeom() = 0;

      /*! returns (part of) a string that allows to identify the
          device-side optix intersect/ch/bounds/etc functions that
          realize this geometry. Eg, if this accel uses macrocells,
          and all the device-functions are named after a scheme
          "<MyScalarType>_MC_<Bounds/Isec/CH/...>()" then this
          function should ask the scalar field type to create
          "<MyScalarType>", and append "_MC" to it */
      virtual std::string getTypeString() const
      { return sf->getTypeString(); }
      
      OWLGeom      geom = 0;
      OWLGroup     group = 0;
      const char  *const ptxCode;
    };
  };
  
  template<typename SF>
  struct MCVolumeAccel : public OWLVolumeAccel<SF>
  {
    struct DD : public OWLVolumeAccel<SF>::DD {
      static void addVars(std::vector<OWLVarDecl> &vars, int base)
      {
        // MCGrid::addVars(vars,base+OWL_OFFSETOF(OWLVolumeAccel<SF>::DD,mcGrid)); 
      }
      MCGrid::DD mcGrid;
    };
    struct Host : public OWLVolumeAccel<SF>::Host
    {
      using Inherited = typename OWLVolumeAccel<SF>::Host;
      
      Host(ScalarField *sf, Volume *volume, const char *ptxCode)
        : Inherited(sf,volume,ptxCode),
          mcGrid(sf->devGroup)
      {}

      void setVariables(OWLGeom geom) override
      {
        Inherited::setVariables(geom);
        mcGrid.setVariables(geom);
      }
      
      void createGeom() override
      {
        auto devGroup = sf->devGroup;
        const std::string typeString
          = getTypeString();
        std::cout << "creating owl geom type for barney type '"
                  << typeString << "'" << std::endl;
        OWLGeomType gt = devGroup->getOrCreateGeomTypeFor
          (ptxCode, [&](DevGroup *dg)
          {
            printf("creating geom type ....");
            std::vector<OWLVarDecl> params;
            DD::addVars(params,0);
            OWLGeomType gt = owlGeomTypeCreate
              (dg->owl,OWL_GEOM_USER,
               sizeof(DD),params.data(),params.size());
            OWLModule module = owlModuleCreate(dg->owl,ptxCode);
            const std::string boundsProg = typeString+"_Bounds";
            const std::string isProg = typeString+"_Isec";
            const std::string chProg = typeString+"_CH";
            owlGeomTypeSetBoundsProg(gt,module,boundsProg.c_str());
            owlGeomTypeSetIntersectProg(gt,0,module,isProg.c_str());
            owlGeomTypeSetClosestHit(gt,0,module,chProg.c_str());
            owlBuildPrograms(dg->owl);
            return gt;
          });
        
        geom = owlGeomCreate(devGroup->owl,gt);
        std::cout << "owl geom created, but not yet set" << std::endl;
      }
      
      void build(bool full_rebuild) override
      {
        PING; 
        if (mcGrid.dims.x == 0) {
          std::cout << "first building macro cell grid ..." << std::endl;
          sf->buildMCs(mcGrid);
        }
        Inherited::build(full_rebuild);
          
        // PING; PRINT(full_rebuild);
        // if (!full_rebuild) return;
        
        // if (!geom) {
        //   std::cout << "first building macro cell grid ..." << std::endl;
        //   sf->buildMCs(mcGrid);
        //   std::cout << "now creating the owl geom" << std::endl;
        //   this->createGeom();
        // };
        // std::cout << "#bn.mc: geometry created" << std::endl;
      };
      MCGrid       mcGrid;
      
      using Inherited::sf;
      using Inherited::ptxCode;
      using Inherited::geom;
      using Inherited::getTypeString;
    };
  };
    
  /*! a OWL geometry that creates one optix geom for each macrocell,
      uses refitting to 'hide' in-active ones, and leaves traversal of
      these macrocells to optix's traversal of the individual per-mc
      prims */
  template<typename SF>
  struct MCRTXVolumeAccel : public MCVolumeAccel<SF> {
    struct DD : public MCVolumeAccel<SF>::DD {};
    
    struct Host : public MCVolumeAccel<SF>::Host {
      using Inherited = typename MCVolumeAccel<SF>::Host;
      
      using Inherited::sf;
      using Inherited::geom;
      using Inherited::mcGrid;
      
      Host(ScalarField *sf, Volume *volume, const char *ptxCode)
        : Inherited(sf,volume,ptxCode)
      {}
      /*! builds the string that allows for properly matching optix
          device progs for this type */
      std::string getTypeString() const override
      { return sf->getTypeString()+"_MCRTX"; }
      
      void createGeom() override
      {
        // let parent class create the geometry itself, _we_ then only
        // set the numprims
        Inherited::createGeom();
        
        // this is a *RTXTraverser* for the mcgrid, we have one prim
        // per cell:
        const int primCount = mcGrid.dims.x*mcGrid.dims.y*mcGrid.dims.z;
        PRINT(primCount);
        owlGeomSetPrimCount(geom,primCount);
      }
    };
  };

  /*! a OWL geometry that creates a *SINGLE* optix primitive for the
      entire macro cell grid, then performs DDA traversal within that
      grid */
  template<typename SF>
  struct MCDDAVolumeAccel : public MCVolumeAccel<SF> {
    struct Host : public MCVolumeAccel<SF>::Host {
      Host(ScalarField *sf, Volume *volume, const char *ptxCode)
        : MCVolumeAccel<SF>::Host(sf,volume,ptxCode)
      {}
      void createGeom() override
      { BARNEY_NYI(); }
    };
  };
  
  
  /*! a macro-cell accelerator, built over some
      (template-parameter'ed) type of underlying volume. The volume
      must be able to compute the macro-cells and majorants, and to
      sample; this class will then do the traversal, and provide the
      'glue' to act as a actual barney volume accelerator */
  template<typename FieldSampler>
  struct MCAccelerator : public VolumeAccel
  {
    struct DD {
      inline __device__
      vec4f sampleAndMap(vec3f P, bool dbg=false) const
      { return volume.sampleAndMap(sampler,P,dbg); }

      /*! our own macro-cell grid to be traversed */
      MCGrid::DD                mcGrid;
      
      /*! whatever the field sampler brings in to be able to sample
          the underlying field */
      typename FieldSampler::DD sampler;
      
      /*! the volume's device data that maps field samples to rgba
          values */
      VolumeAccel::DD           volume;
    };

    MCAccelerator(ScalarField *field, Volume *volume);
    // void build() override;
    
    OWLGeom      geom = 0;
    MCGrid       mcGrid;
    FieldSampler sampler;
  };


  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  template<typename FieldSampler>
  MCAccelerator<FieldSampler>::MCAccelerator(ScalarField *field,
                                              Volume *volume)
    : VolumeAccel(field, volume),
      sampler(field),
      mcGrid(devGroup)
  {}

}

