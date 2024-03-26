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
    matte.samplerType = samplerType;
    if (samplerType == render::IMAGE1D || samplerType == render::IMAGE2D) {
      matte.sampler.image.image
        = owlTextureGetObject(sampler.image.image->owlTex,deviceID);
      matte.sampler.image.inAttribute = sampler.image.inAttribute;
      matte.sampler.image.inTransform = sampler.image.inTransform;
      matte.sampler.image.inOffset = sampler.image.inOffset;
      matte.sampler.image.outTransform = sampler.image.outTransform;
      matte.sampler.image.outOffset = sampler.image.outOffset;
    }
    else if (samplerType == render::TRANSFORM) {
      matte.sampler.transform.inAttribute = sampler.transform.inAttribute;
      matte.sampler.transform.outTransform = sampler.transform.outTransform;
      matte.sampler.transform.outOffset = sampler.transform.outOffset;
    }
    dd = matte;
  }
 
  bool MatteMaterial::setString(const std::string &member, const std::string &value)
  {
    if (Material::setString(member,value)) return true;
    if (member == "sampler.type")  {
      if (value == "image1D")
        { samplerType = render::IMAGE1D; return true; }
      if (value == "image2D")
        { samplerType = render::IMAGE2D; return true; }
      if (value == "transform")
        { samplerType = render::TRANSFORM; return true; }
      return false;
    }
    return false;
  }

  bool MatteMaterial::setObject(const std::string &member, const Object::SP &value)
  {
    if (Material::setObject(member,value)) return true;
    if (member == "sampler.image.image")
      { sampler.image.image = value->as<Texture>(); return true; }
    return false;
  }

  bool MatteMaterial::set1i(const std::string &member, const int &value) 
  {
    if (Material::set1i(member,value)) return true;
    if (member == "sampler.image.inAttribute") 
      { sampler.image.inAttribute = value; return true; }
    if (member == "sampler.transform.inAttribute") 
      { sampler.transform.inAttribute = value; return true; }
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
    if (member == "sampler.image.inOffset") 
      { sampler.image.inOffset = value; return true; }
    if (member == "sampler.image.outOffset") 
      { sampler.image.outOffset = value; return true; }
    if (member == "sampler.transform.outOffset") 
      { sampler.transform.outOffset = value; return true; }
    return false;
  }
  bool MatteMaterial::set4x4f(const std::string &member, const mat4f &value) 
  {
    if (Material::set4x4f(member,value)) return true;
    if (member == "sampler.image.inTransform") 
      { sampler.image.inTransform = value; return true; }
    if (member == "sampler.image.outTransform") 
      { sampler.image.outTransform = value; return true; }
    if (member == "sampler.transform.outTransform") 
      { sampler.transform.outTransform = value; return true; }
    return false;
  }
}
