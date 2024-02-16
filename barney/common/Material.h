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

#include "barney/Texture.h"
#include "barney/Data.h"

namespace barney {
  
  struct Material : public DataGroupObject {
    typedef std::shared_ptr<Material> SP;
    
    struct DD {
      vec3f baseColor;
      float ior;
      float transmission;
      float roughness;
      float metallic;
      cudaTextureObject_t colorTexture;
      cudaTextureObject_t alphaTexture;
    };

    Material(DataGroup *owner) : DataGroupObject(owner) {}
    virtual ~Material() = default;
    
    void commit() override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    
    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    void set(OWLGeom geom) const;
    
    vec3f baseColor { .5f, .5f, .5f };
    float transmission { 0.f };
    float roughness    { 0.f };
    float metallic     { 0.f };
    float ior          { 1.f };
    Texture::SP colorTexture;
    Texture::SP alphaTexture;
  };

}
