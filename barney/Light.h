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

namespace barney {

  struct DataGroup;
  
  struct Light : public DataGroupObject {
    typedef std::shared_ptr<Light> SP;

    Light(DataGroup *owner) : DataGroupObject(owner) {}

    std::string toString() const override { return "Light<>"; }
    
    static Light::SP create(DataGroup *owner, const std::string &name);
  };


  struct DirectionalLight : public Light {
    DirectionalLight(DataGroup *owner) : Light(owner) {}
    struct DD {
      vec3f direction;
      vec3f radiance;
    };
    
    std::string toString() const override { return "DirectionalLight"; }
    
    bool set3f(const std::string &member, const vec3f &value) override;
    
    vec3f direction;
    vec3f radiance;
  };
};
