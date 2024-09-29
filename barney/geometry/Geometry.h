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
#include "barney/render/Ray.h"
#include "barney/render/HostMaterial.h"
#include "barney/render/HitAttributes.h"
#include "barney/render/GeometryAttributes.h"
#include "barney/render/OptixGlobals.h"
#include "barney/render/HostMaterial.h"

namespace barney {
  
  struct ModelSlot;
  using render::GeometryAttribute;
  using render::GeometryAttributes;
  using render::HostMaterial;
  
  struct Geometry : public SlottedObject {
    typedef std::shared_ptr<Geometry> SP;

    struct DD {

      template<typename InterpolatePerVertex>
      inline __device__
      void setHitAttributes(render::HitAttributes &hit,
                            const InterpolatePerVertex &interpolate,
                            bool dbg=false) const;

      render::GeometryAttributes::DD attributes;
      int materialID;
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

    void setAttributesOn(OWLGeom geom);
    
    /*! get the own context that was used to create this geometry */
    OWLContext getOWL() const;

    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool setData(const std::string &member, const Data::SP &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    
    HostMaterial::SP getMaterial() const;
    void setMaterial(HostMaterial::SP);
    
    std::vector<OWLGeom>  triangleGeoms;
    std::vector<OWLGeom>  userGeoms;
    std::vector<OWLGroup> secondPassGroups;

  private:
    render::HostMaterial::SP material;

    render::GeometryAttributes attributes;
  };

  template<typename InterpolatePerVertex>
  inline __device__
  void Geometry::DD::setHitAttributes(render::HitAttributes &hit,
                                      const InterpolatePerVertex &interpolate,
                                      bool dbg) const
  {
    auto set = [&](float4 &out, const GeometryAttribute::DD &in) {
      switch(in.scope) {
      case GeometryAttribute::INVALID:
        /* nothing - leave default */
        break;
      case GeometryAttribute::CONSTANT:
        out = in.value;
        break;
      case GeometryAttribute::PER_PRIM:
        out = in.fromArray.valueAt(hit.primID);
        break;
      case GeometryAttribute::PER_VERTEX:
        out = interpolate(in);
        break; 
      }
    };
    
    for (int i=0;i<attributes.count;i++) {
      float4     &out = hit.attribute[i];
      const auto &in  = this->attributes.attribute[i];
      set(out,in);
    }
    set(hit.color,this->attributes.colorAttribute);
  }
  
}
