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
  
  /*! embree/ospray "Metal" material */
  struct MetalMaterial : public barney::Material {
    MetalMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~MetalMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "MetalMaterial"; }

    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override {};
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    vec3f eta {1.47f, 0.984f, 0.602f}; // index of refraction
    vec3f k {7.64f, 6.55f, 5.36f}; // index of refraction, imaginary part
    float roughness { .1f }; // in [0, 1]; 0==ideally smooth (mirror)
  };
  
}
