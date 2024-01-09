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
#include "barney/volume/TransferFunction.h"

namespace barney {

  struct DataGroup;
  struct Volume;
  struct ScalarField;
  struct MCGrid;

  typedef std::array<int,4> TetIndices;
  typedef std::array<int,5> PyrIndices;
  typedef std::array<int,6> WedIndices;
  typedef std::array<int,8> HexIndices;

  
  struct VolumeAccel {
    typedef std::shared_ptr<VolumeAccel> SP;

    template<typename _Field>
    struct DD {
      typedef _Field Field;
      static void addVars(std::vector<OWLVarDecl> &vars,size_t base);
      
      inline __device__
      vec4f map(float f, bool dbg=false) const
      { return xf.map(f,dbg); }

      inline __device__
      const typename Field::DD &getField() const { return field; }
      TransferFunction::DD xf;
      typename Field::DD   field;
    };
    
    VolumeAccel(ScalarField *field, //Volume *volume,
                TransferFunction *xf);
    
    virtual void build() = 0;
    
    
    // virtual void setVariables(OWLGeom geom, bool firstTime);

    void setVariables(OWLGeom geom);
  
    const TransferFunction *getXF() const;
    
    ScalarField      *const field;
    TransferFunction *const xf;
    // Volume      *const volume;
    DevGroup    *const devGroup;
  };

  // template<typename Field>
  // void VolumeAccel::DD<Field>::addVarDecls(std::vector<OWLVarDecl> &vars,size_t base)
  // {
  //   Field::DD::addVarDecls(vars,base+OWL_OFFSETOF(DD,field));
  //   TransferFunction::DD::addVarDecls(vars,base+OWL_OFFSETOF(DD,xf));
  // }
  
  
  struct SampleableVolumeAccel {
    template<typename Field>
    struct DD : public VolumeAccel::DD<Field> {
      
      inline __device__
      vec4f sampleAndMap(vec3f point, bool dbg=false) const
      {
        float f = this->field.sample(point,dbg);
        if (isnan(f)) return vec4f(0.f);
        return this->map(f,dbg);
      }
    };
  };
    
  
  // template<typename FieldSampler>
  // struct SampleableVolumeAccel : public VolumeAccel {
  //   // template<typename SampleableField>
  // };

  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public Object
  {
    typedef std::shared_ptr<ScalarField> SP;

    struct DD {
      static void addVarDecls(std::vector<OWLVarDecl> &vars, uint32_t base);
      
      box4f  worldBounds;
    };
    
    ScalarField(DevGroup *devGroup)
      : devGroup(devGroup)
    {}

    OWLContext getOWL() const;
    // virtual std::vector<OWLVarDecl> getVarDecls(uint32_t baseOfs) = 0;

    virtual void setVariables(OWLGeom geom) = 0;
    
    virtual VolumeAccel::SP createAccel(Volume *volume) = 0;
    virtual void buildMCs(MCGrid &macroCells)
    { throw std::runtime_error("this calar field type does not know how to build macro-cells"); }
      
    DevGroup *const devGroup;
    box4f     worldBounds;
  };


  // struct SampleableScalarField : public ScalarField {
  //   struct DD : public ScalarField::DD {
  //   };
  // };
  
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
           ScalarField::SP field);

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
               
    ScalarField::SP  field;
    VolumeAccel::SP  accel;
    TransferFunction xf;

    std::vector<OWLGroup> generatedGroups;
    DevGroup *const devGroup;
  };


/*! DEVICE part of somethign that forms a specific volume implementation */
template<typename SampleableField>
struct VolumeOver {
  struct DD {
    typename SampleableField::DD field;
    TransferFunction::DD xf;

    inline __device__ vec4f sampleAndMap(vec3f P) const
    {
      float f = field.sample(P);
      return xf.map(f);
    }
  };
  static void addVars(std::vector<OWLVarDecl> &vars, int ofs)
  {
    SampleableField::addVars(vars,ofs+OWL_OFFSETOF(DD,field));
    TransferFunction::addVars(vars,ofs+OWL_OFFSETOF(DD,xf));
  }
  
};



  
  template<typename DDType>
  struct VolumeAccelGeomFor : public VolumeAccel
  {
    VolumeAccelGeomFor(ScalarField *field,
                       TransferFunction *xf,
                       const char *ptxCode)
      : VolumeAccel(field,xf),
        traverser(this,DDType::createGeom,ptxCode),
        ptxCode(ptxCode),
        sampler(field)
    { build(); }

    void build()
    {
      sampler.build();
      traverser.build();
    }

    const char *ptxCode;
    typename DDType::TravHost    traverser;
    typename DDType::SamplerHost sampler;
  };


  
  inline VolumeAccel::VolumeAccel(ScalarField *field, TransferFunction *xf)
    : field(field), xf(xf), devGroup(field->devGroup)
  {
    assert(field);
    assert(field->devGroup);
    assert(field->devGroup == volume->devGroup);
  }


  inline const TransferFunction *VolumeAccel::getXF() const { return xf; }
  
}
