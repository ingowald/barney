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

#include "barney/material/host/Matte.h"
#include "barney/material/device/Matte.h"
#include "barney/ModelSlot.h"

namespace barney {

  void MatteMaterial::createDD(DD &dd, int deviceID) const 
  {
    render::Matte::DD matte;
    matte.reflectance = reflectance;
    matte.transformSampler.inAttribute = transformSampler.inAttribute;
    matte.transformSampler.outTransform = transformSampler.outTransform;
    matte.transformSampler.outOffset = transformSampler.outOffset;
    dd = matte;
  }
  
  bool MatteMaterial::set1i(const std::string &member, const int &value) 
  {
    if (Material::set1i(member,value)) return true;
    if (member == "sampler.inAttribute") 
      { transformSampler.inAttribute = value; return true; }
    return false;
  }
  bool MatteMaterial::set3f(const std::string &member, const vec3f &value) 
  {
    if (Material::set3f(member,value)) return true;
    if (member == "reflectance") 
      { reflectance = value; return true; }
    return false;
  }
  bool MatteMaterial::set4f(const std::string &member, const vec4f &value) 
  {
    if (Material::set4f(member,value)) return true;
    if (member == "sampler.outOffset") 
      { transformSampler.outOffset = value; return true; }
    return false;
  }
  bool MatteMaterial::set4x4f(const std::string &member, const mat4f &value) 
  {
    if (Material::set4x4f(member,value)) return true;
    if (member == "sampler.outTransform") 
      { transformSampler.outTransform = value; return true; }
    return false;
  }
}
