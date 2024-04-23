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
  
  struct Geometry : public SlottedObject {
    typedef std::shared_ptr<Geometry> SP;

    struct DD {

      inline __device__ void setHit(Ray &ray);
      
      int       materialID;//Material::DD     material;
      struct {
        BNDataType     type;
        AttributeScope scope;
        void          *ptr;
      } attributes[4];
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

    PODData::SP attribute0;
    PODData::SP attribute1;
    PODData::SP attribute2;
    PODData::SP attribute3;
    PODData::SP attribute4;
  };

}
