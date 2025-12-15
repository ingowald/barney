// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
        color.set(sampler);
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
        color.set(value);
        return true;
      }
      
      return false;
    }
    
    bool AnariMatte::set1f(const std::string &member, const float &value) 
    {
      if (HostMaterial::set1f(member,value)) return true;
      
      if (member == "opacity")
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
    
    bool AnariMatte::set4f(const std::string &member, const vec4f &value) 
    {
      if (HostMaterial::set4f(member,value)) return true;
      
      if (member == "color")
        { color.set(value); return true; }
      
      return false;
    }
  }
}
