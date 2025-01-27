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

#include "barney/material/AnariPBR.h"
#include "barney/material/DeviceMaterial.h"

namespace barney {
  namespace render {
    
    DeviceMaterial AnariPBR::getDD(Device *device) 
    {
      DeviceMaterial dd;
      dd.type = DeviceMaterial::TYPE_AnariPBR;
      // baseColor .make(dd.anariPBR.baseColor, deviceID);
      // emission  .make(dd.anariPBR.emission,  deviceID);
      // metallic  .make(dd.anariPBR.metallic,  deviceID);
      // opacity   .make(dd.anariPBR.opacity,   deviceID);
      // roughness .make(dd.anariPBR.roughness, deviceID);
      // ior       .make(dd.anariPBR.ior,       deviceID);

      // transmission.make(dd.anariPBR.transmission,deviceID);

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
