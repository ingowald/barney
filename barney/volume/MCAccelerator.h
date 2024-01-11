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


  struct MCGridAccel {
    MCGridAccel(VolumeAccel *volume,
                OWLGeom (*createGeom)(MCGridAccel *, const char *),
                const char *ptxCode)
      : volume(volume),
        ptxCode(ptxCode),
        mcGrid(volume->field->devGroup)
    {}

    void prepareInitialBuild()
    {
      // gt = createGT(ptxCode);
      PING;
      volume->field->buildMCs(mcGrid);
      PING;
      geom = createGeom(this,ptxCode);
      PING;
      assert(geom);
      // geom = owlGeomCreate(getOWL(),createGT(ptxCode));
      setVariables(geom);
      // owlGeomSetPrimCount(geom,1);
    }
  
    void build()
    {
      PING;
      if (firstTimeBuild) {
        prepareInitialBuild();
        firstTimeBuild = false;
      }
      mcGrid.computeMajorants(volume->getXF());
      setVariables(geom);
    }

    void setVariables(OWLGeom geom)
    {
      mcGrid.setVariables(geom);
      volume->setVariables(geom);
    }

    bool firstTimeBuild = true;
    MCGrid mcGrid;
    VolumeAccel *volume;
    // OWLGeomType gt = 0;
    OWLGeom geom = 0;
    OWLGeom (*createGeom)(MCGridAccel *, const char *);
    const char *ptxCode;
  };
  

  /*! class that creates a optix user geometry with one prim per
    (non-empty) macro cell, and whose intersection code looks up the
    repsecigve cell, then performs woodcock trackign in said cell;
    calling the respective SampleVolume::sampleAndMap() function for
    each woodcock step */
  template<typename SampleableVolume>
  struct RTXMCTraverserGeom {
    typedef typename SampleableVolume::HostSide SamplerHost;
    using TravHost = MCGridAccel;
  
    struct DD : public VolumeOver<SampleableVolume>::DD {
      MCGrid::DD mcGrid;
    
      inline __device__ void intersect(int primID) const
      {
        vec3i cell(primID,1,2);
        for (float t=0.f;t<1.f;t+= .1f) {
          vec4f sample = this->sampleAndMap(vec3f(cell.x+t));
          if (sample.w > 0.f) return;
        }
      }
    };

    static OWLGeom createGeom(MCGridAccel *mcGridAccel, const char *ptxCode)
    {
      BARNEY_NYI();
    }
    // {
    //   yyy();
    // }
    static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
    {
      VolumeOver<SampleableVolume>::addVars(vars,ofs);
      MCGrid::addVars(vars,ofs+OWL_OFFSETOF(DD,mcGrid));
    }
    static OWLGeomType createGT(DevGroup *devGroup,
                                const char *ptxCode)
    {
      std::string gtTypeName = getProgName();
      std::cout << OWL_TERMINAL_GREEN
                << "creating '" << gtTypeName << "' geometry type"
                << OWL_TERMINAL_DEFAULT << std::endl;

      std::vector<OWLVarDecl> params;
      addVars(params,0);
      
      OWLModule module = owlModuleCreate
        (devGroup->owl,ptxCode);
      OWLGeomType gt = owlGeomTypeCreate
        (devGroup->owl,OWL_GEOM_USER,sizeof(DD),
         params.data(),params.size());
      owlGeomTypeSetBoundsProg(gt,module,
                               (gtTypeName+"_Bounds").c_str());
      owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,
                                  (gtTypeName+"_Isec").c_str());
      owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                               (gtTypeName+"_CH").c_str());
      owlBuildPrograms(devGroup->owl);
    
      return gt;
    }
  
    static std::string getProgName()
    { return "RTXMCTraverserGeom_"+SampleableVolume::getProgName(); }
  
    void build();
  };


  template<typename SampleableVolume>
  struct DDATraverserGeom {
    typedef typename SampleableVolume::HostSide SamplerHost;
    using TravHost = MCGridAccel;

    struct DD : public VolumeOver<SampleableVolume>::DD {
      MCGrid::DD mcGrid;
      
      inline __device__ void intersect(int /*ignoreForThisClass*/) const
      {
        dda3(this->mcGrid,[&](vec3i cell) {
            for (float t=0.f;t<1.f;t+= .1f) {
              vec4f sample = this->sampleAndMap(vec3f(cell.x+t));
              if (sample.w > 0.f) return;
            }
          });
      }
      
    };

    DDATraverserGeom(DevGroup *devGroup) : devGroup(devGroup) {}
    inline OWLContext getOWL() const { return devGroup->owl; }
    
    static OWLGeom createGeom(MCGridAccel *mcGridAccel, const char *ptxCode)
    { BARNEY_NYI(); }
    // {
    //   xxx();
    // }
    static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
    {
      VolumeOver<SampleableVolume>::addVars(vars,ofs);
      MCGrid::addVars(vars,ofs+OWL_OFFSETOF(DD,mcGrid));
    }
    static OWLGeomType createGT(DevGroup *devGroup,
                                const char *ptxCode);
  
    static std::string getProgName()
    { return "DDATraverserGeom_"+SampleableVolume::getProgName(); }
  
    void build();

    DevGroup    *const devGroup;
  };

    
    
  template<typename SampleableVolumeWithMCGrid>
  OWLGeomType DDATraverserGeom<SampleableVolumeWithMCGrid>::createGT(DevGroup *devGroup,
                                                                     const char *ptxCode)
  {
    OWLContext owl = devGroup->owl;//getOWL();
    std::vector<OWLVarDecl> vars;
    addVars(vars,0);
      
    OWLModule module = owlModuleCreate(owl,ptxCode);
    OWLGeomType gt = owlGeomTypeCreate
      (owl,OWL_GEOM_USER,
       sizeof(typename SampleableVolumeWithMCGrid::DD),vars.data(),vars.size());
    std::string progName = getProgName();
    owlGeomTypeSetIntersectProg(gt,0,module,progName.c_str());
    return gt;
  }
    

#if 0
  /*! a macro-cell accelerator, built over some
    (template-parameter'ed) type of underlying volume. The volume
    must be able to compute the macro-cells and majorants, and to
    sample; this class will then do the traversal, and provide the
    'glue' to act as a actual barney volume accelerator */
  struct MCAccelerator : public VolumeAccel
  {
    template<typename SampleableField>
    struct DD : public SampleableVolumeAccel::DD<SampleableField> {
      using SampleableVolumeAccel::DD<SampleableField>::field;
      
      static void addVarDecls(std::vector<OWLVarDecl> &vars,size_t base);

      inline __device__ box3f getCellBounds(vec3i cellID) const;
      inline __device__ box3f getCellBounds(int linearID) const;
      
      /*! our own macro-cell grid to be traversed */
      MCGrid::DD                mcGrid;
    };

    MCAccelerator(Volume *volume);
    
    MCGrid       mcGrid;
  };


  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  // template<typename SampleableField>
  MCAccelerator// <SampleableField>
  ::MCAccelerator(Volume *volume)
    : VolumeAccel(volume),
      // sampler(field),
      mcGrid(devGroup)
  {}

  template<typename SampleableField>
  inline __device__
  box3f MCAccelerator::DD<SampleableField>::getCellBounds(vec3i cellID) const 
  {
    box3f bounds;
    bounds.lower
      = lerp(getBox(this->field.worldBounds),
             vec3f(cellID)*rcp(vec3f(mcGrid.dims)));
    bounds.upper
      = lerp(getBox(this->field.worldBounds),
             vec3f(cellID+vec3i(1))*rcp(vec3f(mcGrid.dims)));
    return bounds;
  }
      
  template<typename SampleableField>
  inline __device__
  box3f MCAccelerator::DD<SampleableField>::getCellBounds(int linearID) const 
  {
    vec3i dims = mcGrid.dims;
    vec3i cellID(linearID % dims.x,
                 (linearID / dims.x) % dims.y,
                 linearID / (dims.x*dims.y));
    return getCellBounds(cellID);
  }
#endif
  
}

