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
    if (type == "envmap")
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
  
  EnvMapLight::EnvMapLight(ModelSlot *owner)
    : Light(owner)
  {
    std::cout << OWL_TERMINAL_YELLOW
              << "#bn: created env-map light"
              << OWL_TERMINAL_DEFAULT << std::endl;
  }
  
  void EnvMapLight::commit()
  {
    content.toWorld.vz = normalize(up);
    content.toWorld.vy = normalize(cross(content.toWorld.vz,direction));
    content.toWorld.vx = normalize(cross(content.toWorld.vy,content.toWorld.vz));
    content.toLocal    = rcp(content.toWorld);
    PING;
    PRINT(up);
    PRINT(direction);
    PRINT(content.toWorld);
  }
  
  bool EnvMapLight::set2i(const std::string &member, const vec2i &value) 
  {
    return false;
  }

  bool EnvMapLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "direction") {
      direction = value;
      return true;
    }
    if (member == "up") {
      up = value;
      return true;
    }
    return false;
  }

  bool EnvMapLight::setObject(const std::string &member, const Object::SP &value) 
  {
    if (member == "texture") {
      this->texture = value->as<Texture>();
      content.texture = texture->owlTex;
      return true;
    }
    return false;
  }

};
