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

#include "barney/material/AnariMatte.h"
#include "barney/material/DeviceMaterial.h"

namespace BARNEY_NS {
  namespace render {
    
    DeviceMaterial AnariMatte::getDD(Device *device) 
    {
      DeviceMaterial dd;
      dd.type = DeviceMaterial::TYPE_AnariMatte;
      dd.anariMatte.color = color.getDD(device);
      return dd;
    }

    /*! need wants to accept: 

      "color" = <Sampler>
     */
    bool AnariMatte::setObject(const std::string &member,
                               const Object::SP &value) 
    {
      if (HostMaterial::setObject(member,value)) return true;

      Sampler::SP sampler = value ? value->as<Sampler>() : Sampler::SP();
      if (member == "color") 
        { color.set(sampler); return true; }
      
      return false;
    }
    
    
    bool AnariMatte::setString(const std::string &member,
                               const std::string &value) 
    {
      if (HostMaterial::setString(member,value)) return true;

      if (member == "color") {
        color.set(value);
        return true;
      }
      
      return false;
    }
    
    bool AnariMatte::set3f(const std::string &member, const vec3f &value) 
    {
      if (HostMaterial::set3f(member,value)) return true;
      
      if (member == "color")
        { color.set(value); return true; }
      
      return false;
    }
    
    bool AnariMatte::set4f(const std::string &member, const vec4f &value) 
    {
      if (HostMaterial::set4f(member,value)) return true;
      
      if (member == "color")
        { color.set(value); return true; }
      
      return false;
    }
  }
}
