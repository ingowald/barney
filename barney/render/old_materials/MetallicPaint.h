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
  
  /*! embree/ospray "MetallicPaint" material */
  struct MetallicPaintMaterial : public barney::Material {
    MetallicPaintMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~MetallicPaintMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "MetallicPaintMaterial"; }

    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override {};
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    float eta = 1.45f;
    float flakeAmount = 0.3f;
    vec3f flakeColor { 0.055f, 0.16f, 0.25f };
    float flakeSpread = 0.025f;
    vec3f baseColor { 0.f, 0.03f, 0.07f };
  };
  
}
