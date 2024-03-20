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
    if (samplerType == render::IMAGE1D) {
      OWLBuffer imageBuffer
        = sampler.image1D.image.data
        ? sampler.image1D.image.data->owl
        : 0;
      matte.sampler.image1D.image.data
          = (const vec4f *)owlBufferGetPointer(imageBuffer,deviceID);
      matte.sampler.image1D.image.width = sampler.image1D.image.width;

      matte.sampler.image1D.inAttribute = sampler.image1D.inAttribute;
      matte.sampler.image1D.inTransform = sampler.image1D.inTransform;
      matte.sampler.image1D.inOffset = sampler.image1D.inOffset;
      matte.sampler.image1D.outTransform = sampler.image1D.outTransform;
      matte.sampler.image1D.outOffset = sampler.image1D.outOffset;
    }
    if (samplerType == render::IMAGE2D) {
      OWLBuffer imageBuffer
        = sampler.image2D.image.data
        ? sampler.image2D.image.data->owl
        : 0;
      matte.sampler.image2D.image.data
          = (const vec4f *)owlBufferGetPointer(imageBuffer,deviceID);
      matte.sampler.image2D.image.width = sampler.image2D.image.width;
      matte.sampler.image2D.image.height = sampler.image2D.image.height;

      matte.sampler.image2D.inAttribute = sampler.image2D.inAttribute;
      matte.sampler.image2D.inTransform = sampler.image2D.inTransform;
      matte.sampler.image2D.inOffset = sampler.image2D.inOffset;
      matte.sampler.image2D.outTransform = sampler.image2D.outTransform;
      matte.sampler.image2D.outOffset = sampler.image2D.outOffset;
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

  bool MatteMaterial::setData(const std::string &member, const Data::SP &value)
  {
    if (Material::setData(member,value)) return true;
    if (member == "sampler.image1D.image.data")
      { sampler.image1D.image.data = value->as<PODData>(); return true; }
    if (member == "sampler.image2D.image.data")
      { sampler.image2D.image.data = value->as<PODData>(); return true; }
    return false;
  }

  bool MatteMaterial::set1i(const std::string &member, const int &value) 
  {
    if (Material::set1i(member,value)) return true;
    if (member == "sampler.image1D.image.width") 
      { sampler.image1D.image.width = value; return true; }
    if (member == "sampler.image1D.inAttribute") 
      { sampler.image1D.inAttribute = value; return true; }
    if (member == "sampler.image2D.image.width") 
      { sampler.image2D.image.width = value; return true; }
    if (member == "sampler.image2D.image.height") 
      { sampler.image2D.image.height = value; return true; }
    if (member == "sampler.image2D.inAttribute") 
      { sampler.image2D.inAttribute = value; return true; }
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
    if (member == "sampler.image1D.inOffset") 
      { sampler.image1D.inOffset = value; return true; }
    if (member == "sampler.image1D.outOffset") 
      { sampler.image1D.outOffset = value; return true; }
    if (member == "sampler.image2D.inOffset") 
      { sampler.image2D.inOffset = value; return true; }
    if (member == "sampler.image2D.outOffset") 
      { sampler.image2D.outOffset = value; return true; }
    if (member == "sampler.transform.outOffset") 
      { sampler.transform.outOffset = value; return true; }
    return false;
  }
  bool MatteMaterial::set4x4f(const std::string &member, const mat4f &value) 
  {
    if (Material::set4x4f(member,value)) return true;
    if (member == "sampler.image1D.inTransform") 
      { sampler.image1D.inTransform = value; return true; }
    if (member == "sampler.image1D.outTransform") 
      { sampler.image1D.outTransform = value; return true; }
    if (member == "sampler.image2D.inTransform") 
      { sampler.image2D.inTransform = value; return true; }
    if (member == "sampler.image2D.outTransform") 
      { sampler.image2D.outTransform = value; return true; }
    if (member == "sampler.transform.outTransform") 
      { sampler.transform.outTransform = value; return true; }
    return false;
  }
}
