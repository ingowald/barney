// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/material/AnariPBR.h"
#include "native/material/DeviceMaterial.h"

namespace BARNEY_NS {
  namespace native {

    AnariPBR::AnariPBR(LDGContext *context)
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
      dd.anariPBR.normal       = normal.getDD(device);
      dd.anariPBR.occlusion    = occlusion.getDD(device);
      dd.anariPBR.specular     = specular.getDD(device);
      dd.anariPBR.specularColor = specularColor.getDD(device);
      dd.anariPBR.clearcoat    = clearcoat.getDD(device);
      dd.anariPBR.clearcoatRoughness = clearcoatRoughness.getDD(device);
      dd.anariPBR.clearcoatNormal = clearcoatNormal.getDD(device);
      dd.anariPBR.thickness    = thickness.getDD(device);
      dd.anariPBR.attenuationDistance = attenuationDistance.getDD(device);
      dd.anariPBR.attenuationColor = attenuationColor.getDD(device);
      dd.anariPBR.sheenColor   = sheenColor.getDD(device);
      dd.anariPBR.sheenRoughness = sheenRoughness.getDD(device);
      dd.anariPBR.iridescence  = iridescence.getDD(device);
      dd.anariPBR.iridescenceIor = iridescenceIor.getDD(device);
      dd.anariPBR.iridescenceThickness = iridescenceThickness.getDD(device);

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
      if (member == "opacity")
        { opacity.set(sampler); return true; }
      if (member == "ior")
        { ior.set(sampler); return true; }
      if (member == "transmission")
        { transmission.set(sampler); return true; }
      if (member == "normal")
        { normal.set(sampler); return true; }
      if (member == "occlusion")
        { occlusion.set(sampler); return true; }
      if (member == "specular")
        { specular.set(sampler); return true; }
      if (member == "specularColor")
        { specularColor.set(sampler); return true; }
      if (member == "clearcoat")
        { clearcoat.set(sampler); return true; }
      if (member == "clearcoatRoughness")
        { clearcoatRoughness.set(sampler); return true; }
      if (member == "clearcoatNormal")
        { clearcoatNormal.set(sampler); return true; }
      if (member == "thickness")
        { thickness.set(sampler); return true; }
      if (member == "attenuationDistance")
        { attenuationDistance.set(sampler); return true; }
      if (member == "attenuationColor")
        { attenuationColor.set(sampler); return true; }
      if (member == "sheenColor")
        { sheenColor.set(sampler); return true; }
      if (member == "sheenRoughness")
        { sheenRoughness.set(sampler); return true; }
      if (member == "iridescence")
        { iridescence.set(sampler); return true; }
      if (member == "iridescenceThickness")
        { iridescenceThickness.set(sampler); return true; }
      // "emissive" is the ANARI spec name; accept it as the emission map.
      if (member == "emissive" || member == "emission")
        { emission.set(sampler); return true; }
      
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
      if (member == "opacity")
        { opacity.set(value); return true; }
      if (member == "normal")
        { normal.set(value); return true; }
      if (member == "clearcoatNormal")
        { clearcoatNormal.set(value); return true; }
      if (member == "occlusion")
        { occlusion.set(value); return true; }
      if (member == "specular")
        { specular.set(value); return true; }
      if (member == "specularColor")
        { specularColor.set(value); return true; }
      if (member == "clearcoat")
        { clearcoat.set(value); return true; }
      if (member == "clearcoatRoughness")
        { clearcoatRoughness.set(value); return true; }
      if (member == "thickness")
        { thickness.set(value); return true; }
      if (member == "attenuationDistance")
        { attenuationDistance.set(value); return true; }
      if (member == "attenuationColor")
        { attenuationColor.set(value); return true; }
      if (member == "sheenColor")
        { sheenColor.set(value); return true; }
      if (member == "sheenRoughness")
        { sheenRoughness.set(value); return true; }
      if (member == "iridescence")
        { iridescence.set(value); return true; }
      if (member == "iridescenceIor")
        { iridescenceIor.set(value); return true; }
      if (member == "iridescenceThickness")
        { iridescenceThickness.set(value); return true; }
      // "emissive" is the ANARI spec name; accept it as the emission map.
      if (member == "emissive" || member == "emission")
        { emission.set(value); return true; }
      
      return false;
    }
    
    bool AnariPBR::set1f(const std::string &member, const float &value) 
    {
      if (HostMaterial::set1f(member,value)) return true;
      
      if (member == "metallic")
        { metallic.set(value); return true; }
      if (member == "roughness")
        { roughness.set(value); return true; }
      if (member == "ior")
        { ior.set(value); return true; }
      if (member == "transmission")
        { transmission.set(value); return true; }
      if (member == "opacity")
        { opacity.set(value); return true; }
      if (member == "specular")
        { specular.set(value); return true; }
      if (member == "clearcoat")
        { clearcoat.set(value); return true; }
      if (member == "clearcoatRoughness")
        { clearcoatRoughness.set(value); return true; }
      if (member == "thickness")
        { thickness.set(value); return true; }
      if (member == "attenuationDistance")
        { attenuationDistance.set(value); return true; }
      if (member == "sheenRoughness")
        { sheenRoughness.set(value); return true; }
      if (member == "iridescence")
        { iridescence.set(value); return true; }
      if (member == "iridescenceIor")
        { iridescenceIor.set(value); return true; }
      if (member == "iridescenceThickness")
        { iridescenceThickness.set(value); return true; }
      if (member == "occlusion")
        { occlusion.set(value); return true; }
      
      return false;
    }
    
    bool AnariPBR::set3f(const std::string &member, const vec3f &value) 
    {
      if (HostMaterial::set3f(member,value)) return true;
      
      if (member == "baseColor")
        { baseColor.set(value); return true; }
      if (member == "specularColor")
        { specularColor.set(value); return true; }
      if (member == "attenuationColor")
        { attenuationColor.set(value); return true; }
      if (member == "sheenColor")
        { sheenColor.set(value); return true; }
      if (member == "normal")
        { normal.set(value); return true; }
      // "emissive" is the ANARI spec name; accept it as the emission color.
      if (member == "emissive" || member == "emission")
        { emission.set(value); return true; }
        
      return false;
    }

    bool AnariPBR::set4f(const std::string &member, const vec4f &value) 
    {
      if (HostMaterial::set4f(member,value)) return true;
      
      if (member == "baseColor")
        { baseColor.set(value); return true; }
      // "emissive" is the ANARI spec name; accept it as the emission color.
      if (member == "emissive" || member == "emission")
        { emission.set(value); return true; }
        
      return false;
    }
  }
}
