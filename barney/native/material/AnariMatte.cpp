// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/material/AnariMatte.h"
#include "native/material/DeviceMaterial.h"

namespace BARNEY_NS {
  namespace native {

    AnariMatte::AnariMatte(LDGContext *context)
      : HostMaterial(context)
    {}
    
    DeviceMaterial AnariMatte::getDD(Device *device) 
    {
      DeviceMaterial dd;
      dd.type = DeviceMaterial::TYPE_AnariMatte;
      packCoverage(dd, opacity.isConstantScalar(1.f) && color.isConstantAlpha(1.f));
      dd.anariMatte.color = color.getDD(device);
      dd.anariMatte.opacity = opacity.getDD(device);
      return dd;
    }

    bool AnariMatte::setObject(const std::string &member,
                               const Object::SP &value) 
    {
      if (HostMaterial::setObject(member,value)) return true;

      if (member == "color") {
        Sampler::SP sampler = value ? value->as<Sampler>() : Sampler::SP();
        color.set(sampler);
        return true;
      }
      if (member == "opacity") {
        Sampler::SP sampler = value ? value->as<Sampler>() : Sampler::SP();
        opacity.set(sampler);
        return true;
      }
      
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
      
      if (member == "opacity") {
        opacity.set(value);
        return true;
      }
      
      return false;
    }

    
    bool AnariMatte::set1f(const std::string &member, const float &value) 
    {
      if (HostMaterial::set1f(member,value)) return true;
      
      if (member == "opacity")
        { opacity.set(value); return true; }
      
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
