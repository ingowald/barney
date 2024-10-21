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

#pragma once

#include <array>

#include "barney/Object.h"
#include "barney/volume/TransferFunction.h"
#include "barney/volume/ScalarField.h"

namespace barney {

  struct ModelSlot;
  struct Volume;
  struct ScalarField;
  struct MCGrid;
  
  typedef std::array<int,4> TetIndices;
  typedef std::array<int,5> PyrIndices;
  typedef std::array<int,6> WedIndices;
  typedef std::array<int,8> HexIndices;

  /*! helper class that performs woodcock sampling over a given
      parameter range, for a given sample'able volume type */
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
        // if (dbg) printf("sample at t %f -> %f %f %f : %f\n",
        //                 t,sample.x,
        //                 sample.y,
        //                 sample.z,
        //                 sample.w);
        if (sample.w >= rand()*majorant) {
          tRange.upper = t;
          return true;
        }
      }
    }
  };
  
  struct VolumeAccel {
    /*! one particular problem of _volume_ accels is that due to
        changes to the transfer function the number of 'valid'
        (owl-)prims in a given group can change over successive
        builds. one option to handle that is to always rebuild
        everything form scratch, but that is expensive. Instead, we
        have this function to allow a given volume to specify whether
        it wants a full rebuild (either because it doesn't have nay
        special pass for refitting, or because it's the first time
        this thing is ever built, etc - UpdateMode::FULL_REBUILD), or
        whether it is fine with refitting (UpdateMode::REFIT)). The
        third update mode - "BUILD_THEN_REFIT" - allows a volume accel
        to request a full rebuild _and_ a additional refit after that;
        this is in order to allow enabling all prims in the initial
        build, and then disabling all majorant-zero ones in a second
        pass. 

        Note: If any one of the voluems in a group request either
        full_rebuild or build_then_refit, then all prims will have
        their 'build(fullRebuild)' method called with rebuild==true;
        if any one of the volumes in a group request either refit or
        build_then_refit, then build will be called (possibly a second
        time!) with fullRebuild==false. In particular, this does mean
        that some geometries will have their build called twice - once
        with fullRebuild true, once with false - even if _they_ have
        not asked for that kind of pass */
    typedef enum { FULL_REBUILD, BUILD_THEN_REFIT, REFIT,
      HAS_ITS_OWN_GROUP } UpdateMode;
    
    typedef std::shared_ptr<VolumeAccel> SP;

    /*! device data for a volume accel - takes the device data for the
        underlying scalar field, and 'adds' a transfer function (and
        then gets the ability to sample field and map with xf */
    template<typename ScalarFieldSampler>
    struct DD : public ScalarFieldSampler::DD {
      using Inherited = typename ScalarFieldSampler::DD;
      
      inline __device__
      vec4f sampleAndMap(vec3f point, bool dbg=false) const
      {
        float f = this->sample(point,dbg);
        if (isnan(f)) return vec4f(0.f);
        vec4f mapped = xf.map(f,dbg);
        return Inherited::mapColor(mapped,point,f);
      }

      static void addVars(std::vector<OWLVarDecl> &vars, int base)
      {
        Inherited::addVars(vars,base);
        TransferFunction::DD::addVars(vars,base+OWL_OFFSETOF(DD,xf));
      }
      
      TransferFunction::DD xf;
    };
    
    VolumeAccel(ScalarField *sf, Volume *volume);

    virtual void setVariables(OWLGeom geom);
    
    virtual UpdateMode updateMode()
    { return FULL_REBUILD; }

    virtual void build(bool full_rebuild) = 0;

    OWLContext getOWL() const;
    const TransferFunction *getXF() const;
    
    ScalarField *const sf = 0;
    Volume      *const volume = 0;
    DevGroup    *const devGroup = 0;
  };
  
  /*! a *volume* is a scalar field with a transfer function applied to
      it; it's main job is to create something that can intersect a
      ray with that scalars-plus-transferfct thingy, for which it will
      use some kind of volume accelerator that implements the
      scalar-field type specific stuff (eg, traverse a bvh over
      elements, or look up a 3d texture, etc) */
  struct Volume : public Object
  {
    typedef std::shared_ptr<Volume> SP;
    
    Volume(DevGroup *devGroup,
           ScalarField::SP sf);

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Volume{}"; }

    VolumeAccel::UpdateMode updateMode() { return accel->updateMode(); }
    
    /*! (re-)build the accel structure for this volume, probably after
        changes to transfer functoin (or later, scalar field) */
    virtual void build(bool full_rebuild);
    
    void setXF(const range1f &domain,
               const std::vector<vec4f> &values,
               float baseDensity)
    { xf.set(domain,values,baseDensity); }
               
    ScalarField::SP  sf;
    VolumeAccel::SP  accel;
    TransferFunction xf;

    std::vector<OWLGroup> generatedGroups;
    DevGroup *const devGroup;
  };


  inline VolumeAccel::VolumeAccel(ScalarField *sf, Volume *volume)
    : sf(sf), volume(volume), devGroup(sf->devGroup)
  {
    assert(sf);
    assert(volume);
    assert(sf->devGroup);
  }
}
