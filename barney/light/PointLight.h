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

#include "barney/light/Light.h"

namespace barney {

  struct PointLight : public Light {
    struct DD : public Light::DD {
      vec3f position;
      float intensity;
      float power;
    };
    
    typedef std::shared_ptr<PointLight> SP;
    PointLight(Context *context, int slot) : Light(context,slot) {}
    
    DD getDD(const affine3f &instanceXfm) const;
    
    std::string toString() const override { return "PointLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    vec3f position = vec3f(0.f,0.f,0.f);
    float power = 1.f;
    float intensity = NAN;
  };

}
