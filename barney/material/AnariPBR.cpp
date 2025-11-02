// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/material/AnariPBR.h"
#include "barney/material/DeviceMaterial.h"

namespace BARNEY_NS {
  namespace render {

    AnariPBR::AnariPBR(SlotContext *context)
      : HostMaterial(context)
    {}
    
    DeviceMaterial AnariPBR::getDD(Device *device) 
    {
      DeviceMaterial dd;
      dd.type = DeviceMaterial::TYPE_AnariPBR;

      dd.anariPBR.baseColor    = baseColor.getDD(device);
      dd.anariPBR.emission     = emission.getDD(device);
      dd.anariPBR.metallic     = metallic.getDD(device);
      dd.anariPBR.opacity      = opacity.getDD(device);
      dd.anariPBR.roughness    = roughness.getDD(device);
      dd.anariPBR.ior          = ior.getDD(device);
      dd.anariPBR.transmission = transmission.getDD(device);

      return dd;
    }
    
    bool AnariPBR::setObject(const std::string &member, const Object::SP &value) 
    {
      if (HostMaterial::setObject(member,value)) return true;
      
      Sampler::SP sampler = value ? value->as<Sampler>() : Sampler::SP();
      if (member == "baseColor") 
        { baseColor.set(sampler); return true; }
      if (member == "metallic") 
        { metallic.set(sampler); return true; }
      if (member == "roughness") 
        { roughness.set(sampler); return true; }
      
      return false;
    }
    
    bool AnariPBR::setString(const std::string &member, const std::string &value) 
    {
      if (HostMaterial::setString(member,value)) return true;

      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "metallic")
        { metallic.set(value); return true; }
      if (member == "roughness")
        { roughness.set(value); return true; }
      if (member == "ior")
        { ior.set(value); return true; }
      if (member == "transmission")
        { transmission.set(value); return true; }
      
      return false;
    }
    
    bool AnariPBR::set1f(const std::string &member, const float &value) 
    {
      if (HostMaterial::set1f(member,value)) return true;
      
      if (member == "metallic")
        { metallic.set(value); return true; }
      if (member == "roughness")
        { roughness.set(clamp(value,.2f,1.f)); return true; }
      if (member == "ior")
        { ior.set(value); return true; }
      if (member == "transmission")
        { transmission.set(value); return true; }
      if (member == "opacity")
        { opacity.set(value); return true; }
      if (member == "specular")
        { /* IGNORE FOR NOW */return true; }
      
      return false;
    }
    
    bool AnariPBR::set3f(const std::string &member, const vec3f &value) 
    {
      if (HostMaterial::set3f(member,value)) return true;
      
      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "emission")
        { emission.set(value); return true; }
      if (member == "specularColor")
        { /* IGNORE FOR NOW */return true; }
      if (member == "emissive")
        { /* IGNORE FOR NOW */return true; }
        
      return false;
    }

    bool AnariPBR::set4f(const std::string &member, const vec4f &value) 
    {
      if (HostMaterial::set4f(member,value)) return true;
      
      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "emission")
        { emission.set(value); return true; }
      if (member == "specularColor")
        { /* IGNORE FOR NOW */return true; }
      if (member == "emissive")
        { /* IGNORE FOR NOW */return true; }
        
      return false;
    }
  }
}
