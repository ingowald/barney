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

#include "barney/material/host/Material.h"

namespace barney {
  
  /*! embree/ospray "Matte" material */
  struct MatteMaterial : public barney::Material {
    MatteMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~MatteMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "MatteMaterial"; }

    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override {};
    bool setString(const std::string &member, const std::string &value) override;
    bool setData(const std::string &member, const Data::SP &value) override;
    bool set1i(const std::string &member, const int &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool set4f(const std::string &member, const vec4f &value) override;
    bool set4x4f(const std::string &member, const mat4f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    vec3f reflectance { 0.55f, 0.0f, 0.0f };
    render::SamplerType samplerType{render::NO_SAMPLER};
    struct {
      struct {
        int inAttribute { 0 };
        mat4f inTransform { mat4f::identity() };
        vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
        mat4f outTransform { mat4f::identity() };
        vec4f outOffset { 0.f, 0.f, 0.f, 0.f };
        struct {
          PODData::SP data;
          int width { 0 };
          render::WrapMode wrapMode{render::CLAMP};
        } image;
      } image1D;
      struct {
        int inAttribute { 0 };
        mat4f inTransform { mat4f::identity() };
        vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
        mat4f outTransform { mat4f::identity() };
        vec4f outOffset { 0.f, 0.f, 0.f, 0.f };
        struct {
          PODData::SP data;
          int width { 0 };
          int height { 0 };
          render::WrapMode wrapMode1{render::CLAMP};
          render::WrapMode wrapMode2{render::CLAMP};
        } image;
      } image2D;
      struct {
        int inAttribute { 0 };
        mat4f outTransform;
        vec4f outOffset { 0 };
      } transform;
    } sampler;
  };
  
}
