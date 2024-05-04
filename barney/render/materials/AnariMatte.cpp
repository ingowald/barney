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

#include "barney/render/materials/AnariMatte.h"
#include "barney/render/DeviceMaterial.h"

namespace barney {
  namespace render {
    
    void AnariMatte::createDD(DeviceMaterial &dd, int deviceID) const 
    {
      dd.type = DeviceMaterial::TYPE_AnariMatte;
      color.make(dd.anariMatte.color,deviceID);
    }
    
    bool AnariMatte::setString(const std::string &member, const std::string &value) 
    {
      PRINT(member); PRINT(value);
      if (HostMaterial::setString(member,value)) return true;

      if (member == "color") 
        { color.set(value); return true; }
      
      return false;
    }
    
    bool AnariMatte::set3f(const std::string &member, const vec3f &value) 
    {
      if (HostMaterial::set3f(member,value)) return true;
      
      if (member == "color")
        { color.set(value); return true; }
      
      return false;
    }
  }
}
