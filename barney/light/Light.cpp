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

#include "barney/light/Light.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace barney {

  Light::SP Light::create(ModelSlot *owner,
                          const std::string &type)
  {
    if (type == "directional")
      return std::make_shared<DirLight>(owner);
    if (type == "quad")
      return std::make_shared<QuadLight>(owner);
    if (type == "environment")
      return std::make_shared<EnvMapLight>(owner);
    
    owner->context->warn_unsupported_object("Light",type);
    return {};
  }

  // ==================================================================
  
  bool DirLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "direction") {
      content.direction = normalize(value);
      return true;
    }
    if (member == "radiance") {
      content.radiance = value;
      return true;
    }
    return false;
  }

  // ==================================================================
  
  bool QuadLight::set3f(const std::string &member, const vec3f &value) 
  {
    return false;
  }

  // ==================================================================
  
  void EnvMapLight::commit()
  {
  }
  
  bool EnvMapLight::set2i(const std::string &member, const vec2i &value) 
  {
    return false;
  }

  bool EnvMapLight::set3f(const std::string &member, const vec3f &value) 
  {
    return false;
  }

  bool EnvMapLight::set4x3f(const std::string &member, const affine3f &value) 
  {
    if (member == "envMap.transform") {
      content.transform = value;
      PING; PRINT(content.transform);
      return true;
    }
    return false;
  }

  bool EnvMapLight::setObject(const std::string &member, const Object::SP &value) 
  {
    if (member == "envMap.texture") {
      this->texture = value->as<Texture>();
      content.texture = texture->owlTex;
      return true;
    }
    return false;
  }

};
