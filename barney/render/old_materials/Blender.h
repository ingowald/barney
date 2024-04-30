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
  
  /*! embree/ospray "Blender" material */
  struct BlenderMaterial : public barney::Material {
    BlenderMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~BlenderMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "BlenderMaterial"; }

    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override {};
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    // vec3f baseColor              = { .8f, .8f, .8f };

    vec3f base_color = vec3f(.8, .8, .8);
    vec3f subsurface_radius = vec3f(1.0, .2, .1);
    vec3f subsurface_color = vec3f(.8, .8, .8);
    float subsurface = 0.0;
    float metallic = 0.0;
    float specular = .5;
    float specular_tint = 0.0;
    float roughness = .5;
    float anisotropic = 0.0;
    float anisotropic_rotation = 0.0;
    float sheen = 0.0;
    float sheen_tint = 0.5;
    float clearcoat = 0.0;
    float clearcoat_roughness = .03f;
    float ior = 1.45f;
    float transmission = 0.0;
    float transmission_roughness = 0.0;
  };
  
}
