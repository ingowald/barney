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

#include "barney/material/host/Glass.h"
#include "barney/material/device/Glass.h"
#include "barney/ModelSlot.h"

namespace barney {

  inline vec3f log(vec3f v)
  {
    return vec3f(logf(v.x),
                 logf(v.y),
                 logf(v.z));
  }
  
  void GlassMaterial::createDD(DD &dd, int deviceID) const 
  {
    render::Glass::DD self;
    self.mediumInside.ior = etaInside;
    self.mediumInside.attenuation = log(attenuationColorInside)/attenuationDistance;
    self.mediumOutside.ior = etaOutside;
    self.mediumOutside.attenuation = log(attenuationColorOutside)/attenuationDistance;
    // printf("eta %f %f\n",etaInside,etaOutside);
    dd = self;
  }
  
  bool GlassMaterial::set1f(const std::string &member, const float &value) 
  {
    if (Material::set1f(member,value)) return true;
    if (member == "etaInside") 
      { etaInside = value; return true; }
    if (member == "etaOutside") 
      { etaOutside = value; return true; }
    if (member == "attenuationDistance") 
      { attenuationDistance = value; return true; }
    return false;
  }
  
  bool GlassMaterial::set3f(const std::string &member, const vec3f &value) 
  {
    if (Material::set3f(member,value)) return true;
    if (member == "attenuationColorInside") 
      { attenuationColorInside = value; return true; }
    if (member == "attenuationColorOutside") 
      { attenuationColorOutside = value; return true; }
    return false;
  }
}
