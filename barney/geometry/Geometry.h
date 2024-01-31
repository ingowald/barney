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

#include "barney/Object.h"
#include "barney/Ray.h"
#include "barney/Texture.h"

namespace barney {

  struct DataGroup;
  
  struct Material {
    struct DD {
      vec3f baseColor;
      cudaTextureObject_t colorTexture;
      cudaTextureObject_t alphaTexture;
    };
    vec3f baseColor;
    Texture::SP colorTexture;
    Texture::SP alphaTexture;
  };

  struct Geometry : public Object {
    typedef std::shared_ptr<Geometry> SP;

    struct DD {
      Material::DD     material;
    };
    
    Geometry(DataGroup *owner,
             const Material &material)
      : owner(owner),
        material(material)
    {}

    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Geometry{}"; }

    /*! ask this geometry to build whatever owl geoms it needs to build */
    virtual void build() {}
    
    void setMaterial(OWLGeom geom);
    
    /*! get the own context that was used to create this geometry */
    OWLContext getOWL() const;
    
    std::vector<OWLGeom>  triangleGeoms;
    std::vector<OWLGeom>  userGeoms;
    std::vector<OWLGroup> secondPassGroups;
    
    Material    material;
    DataGroup  *owner;
  };

}
