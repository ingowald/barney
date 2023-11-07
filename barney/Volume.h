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

#include <array>

#include "barney/Object.h"
// todo: move to barney/volume/TransferFunction :
#include "barney/unstructured/TransferFunction.h"
namespace barney {

  struct DataGroup;
  struct Volume;
  struct ScalarField;
  
  typedef std::array<int,4> TetIndices;
  typedef std::array<int,5> PyrIndices;
  typedef std::array<int,6> WedIndices;
  typedef std::array<int,8> HexIndices;

  struct VolumeAccel {
    typedef std::shared_ptr<VolumeAccel> SP;

    struct DD {
      template<typename FieldSampler>
      inline __device__
      vec4f sampleAndMap(const FieldSampler &field,
                         vec3f point, bool dbg=false) const
      {
        float f = field.sample(point,dbg);
        if (isnan(f)) return vec4f(0.f);
        return xf.map(f,dbg);
      }
      TransferFunction::DD xf;
    };
    
    VolumeAccel(ScalarField *field, Volume *volume);
    
    virtual void build() = 0;
    
    ScalarField *const field;
    Volume      *const volume;
    DevGroup    *const devGroup;
  };

  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public Object
  {
    typedef std::shared_ptr<ScalarField> SP;
    
    ScalarField(DevGroup *devGroup)
      : devGroup(devGroup)
    {}

    OWLContext getOWL() const;
    
    virtual VolumeAccel::SP createAccel(Volume *volume) = 0;
    
    DevGroup    *const devGroup;
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

    /*! (re-)build the accel structure for this volume, probably after
        changes to transfer functoin (or later, scalar field) */
    virtual void build();
    
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


  inline VolumeAccel::VolumeAccel(ScalarField *field, Volume *volume)
    : field(field), volume(volume), devGroup(field->devGroup)
  {
    assert(field);
    assert(volume);
    assert(field->devGroup);
    assert(field->devGroup == volume->devGroup);
  }
}
