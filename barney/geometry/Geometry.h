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

#include "barney/Object.h"
#include "barney/Ray.h"
#include "barney/common/Texture.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/material/host/Material.h"

namespace barney {

  struct ModelSlot;

  struct OptixGlobals : public render::HitAttributes::Globals {
    static inline __device__ const OptixGlobals &get();
  };
  
  __constant__ OptixGlobals optixLaunchParams;
  
  inline __device__ const OptixGlobals &OptixGlobals::get() { return optixLaunchParams; }
  
  
  struct Geometry : public SlottedObject {
    typedef std::shared_ptr<Geometry> SP;

    struct DD {

      template<typename InterpolatePerVertex>
      inline __device__
      void evalAttributesAndStoreHit(Ray &ray,
                                     render::HitAttributes &hit,
                                     const InterpolatePerVertex &interpolate) const;

      int                       materialID;
      render::GeometryAttribute::DD attributes[render::numAttributes];
    };
    
    Geometry(ModelSlot *owner);
    virtual ~Geometry();

    static Geometry::SP create(ModelSlot *dg, const std::string &type);
    
    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Geometry{}"; }

    /*! ask this geometry to build whatever owl geoms it needs to build */
    virtual void build() {}
    
    /*! get the own context that was used to create this geometry */
    OWLContext getOWL() const;

    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool setData(const std::string &member, const Data::SP &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    
    std::vector<OWLGeom>  triangleGeoms;
    std::vector<OWLGeom>  userGeoms;
    std::vector<OWLGroup> secondPassGroups;
    
    Material::SP material;

    render::GeometryAttribute::OnHost attribute[render::numAttributes];
    // PODData::SP attribute0;
    // PODData::SP attribute1;
    // PODData::SP attribute2;
    // PODData::SP attribute3;
    // PODData::SP attribute4;
  };

  template<typename InterpolatePerVertex>
  inline __device__
  void Geometry::DD::evalAttributesAndStoreHit(Ray &ray,
                                               render::HitAttributes &hit,
                                               const InterpolatePerVertex &interpolate)
    const
  {
    using namespace render;
    const DeviceMaterial &material  = hit.globals.deviceMaterials[this->materialID];
    const PackedBSDF      bsdf      = material.createBSDF(hit);
    ray.setHit(hit.worldPosition,hit.worldNormal,hit.t,bsdf);
    
    {
      for (int i=0;i<numAttributes;i++) {
        const GeometryAttribute::DD &attrib = this->attributes[i];
        if (attrib.scope == GeometryAttribute::CONSTANT)
          hit.attribute[i] = attrib.value;
        else if (attrib.scope == GeometryAttribute::PER_PRIM)
          hit.attribute[i] = attrib.fromArray.valueAt(hit.primID);
        else if (attrib.scope == GeometryAttribute::PER_VERTEX)
          hit.attribute[i] = interpolate(attrib);
        else 
          /* undefined attrib; nothing to do, leave on default */;
      }
    }
  }
  
}
