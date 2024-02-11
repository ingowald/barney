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

#include "barney/Light.h"
#include "barney/DataGroup.h"
#include "barney/Context.h"

namespace barney {

  Light::SP Light::create(DataGroup *owner,
                          const std::string &type)
  {
    if (type == "directional")
      return std::make_shared<DirectionalLight>(owner);
    
    owner->context->warn_unsupported_object("Light",type);
    return {};
  }

  bool DirectionalLight::set(const std::string &member, const vec3f &value) 
  {
    if (member == "direction") {
      direction = value;
      return true;
    }
    if (member == "radiance") {
      radiance = value;
      return true;
    }
    return 0;
  }

};
