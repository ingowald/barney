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
#include "barney/Data.h"
#include "barney/Texture.h"
#include "barney/render/World.h"

namespace barney {

  struct DataGroup;
  
  struct Light : public DataGroupObject {
    typedef std::shared_ptr<Light> SP;

    Light(DataGroup *owner) : DataGroupObject(owner) {}

    std::string toString() const override { return "Light<>"; }
    
    static Light::SP create(DataGroup *owner, const std::string &name);
  };


  struct EnvMapLight : public Light {
    typedef std::shared_ptr<EnvMapLight> SP;
    EnvMapLight(DataGroup *owner) : Light(owner) {}
    
    std::string toString() const override { return "EnvMapLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set2i(const std::string &member, const vec2i &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool set4x3f(const std::string &member, const affine3f &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    render::EnvMapLight content;
    Texture::SP texture;
  };

  /*! TODO: this currently sets variables directly, without commit ... */
  struct DirLight : public Light {
    typedef std::shared_ptr<DirLight> SP;
    DirLight(DataGroup *owner) : Light(owner) {}
    
    std::string toString() const override { return "DirLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    render::DirLight content;
  };

  /*! TODO: this currently sets variables directly, without commit ... */
  struct QuadLight : public Light {
    typedef std::shared_ptr<QuadLight> SP;
    QuadLight(DataGroup *owner) : Light(owner) {}

    std::string toString() const override { return "DirectionalLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    render::QuadLight content;
  };
};
