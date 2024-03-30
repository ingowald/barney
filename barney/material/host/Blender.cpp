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

#include "barney/material/host/Blender.h"
#include "barney/material/device/Blender.h"
#include "barney/ModelSlot.h"

namespace barney {

  void BlenderMaterial::createDD(DD &dd, int deviceID) const 
  {
    render::Blender::DD blender;

#define SET(a) blender.a = a
    SET(base_color);
    SET(subsurface_radius);
    SET(subsurface_color);
    SET(subsurface);
    SET(metallic);
    SET(specular);
    SET(specular_tint);
    SET(roughness);
    SET(anisotropic);
    SET(anisotropic_rotation);
    SET(sheen);
    SET(sheen_tint);
    SET(clearcoat);
    SET(clearcoat_roughness);
    SET(ior);
    SET(transmission);
    SET(transmission_roughness);
#undef SET
    dd = blender;
  }
  
  bool BlenderMaterial::set3f(const std::string &member, const vec3f &value) 
  {
    if (Material::set3f(member,value)) return true;
    if (member == "baseColor") 
      { base_color = value; return true; }
    return false;
  }

  bool BlenderMaterial::set1f(const std::string &member, const float &value) 
  {
    if (Material::set3f(member,value)) return true;
    if (member == "subsurface") 
      { subsurface = value; return true; }
    if (member == "metallic") 
      { metallic = value; return true; }
    if (member == "specular") 
      { specular = value; return true; }
    if (member == "specular_tint") 
      { specular_tint = value; return true; }
    if (member == "roughness") 
      { roughness = value; return true; }
    if (member == "anisotropic") 
      { anisotropic = value; return true; }
    if (member == "anisotropic_rotation") 
      { anisotropic_rotation = value; return true; }
    if (member == "sheen") 
      { sheen = value; return true; }
    if (member == "sheen_tint") 
      { sheen_tint = value; return true; }
    if (member == "clearcoat") 
      { clearcoat = value; return true; }
    if (member == "clearcoat_roughness") 
      { clearcoat_roughness = value; return true; }
    if (member == "ior") 
      { ior = value; return true; }
    if (member == "transmission") 
      { ior = transmission; return true; }
    if (member == "transmission_roughness") 
      { ior = transmission_roughness; return true; }
    return false;
  }
}
