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

#include "barney/material/host/Plastic.h"
#include "barney/material/device/Plastic.h"
#include "barney/ModelSlot.h"

namespace barney {

  void PlasticMaterial::createDD(DD &dd, int deviceID) const 
  {
    render::Plastic::DD plastic;
    plastic.eta = eta;
    plastic.roughness = roughness;
    plastic.pigmentColor = pigmentColor;
    dd = plastic;
  }
  
  bool PlasticMaterial::set1f(const std::string &member, const float &value) 
  {
    if (Material::set1f(member,value)) return true;
    if (member == "eta") 
      { eta = value; return true; }
    if (member == "roughness") 
      { roughness = value; return true; }
    return false;
  }
  
  bool PlasticMaterial::set3f(const std::string &member, const vec3f &value) 
  {
    if (Material::set3f(member,value)) return true;
    if (member == "pigmentColor") 
      { pigmentColor = value; return true; }
    return false;
  }
}
