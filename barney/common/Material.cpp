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

#include "barney/common/Material.h"
#include "barney/DataGroup.h"

namespace barney {

  Material::SP Material::create(DataGroup *dg, std::string &type)
  {
    // iw - "eventually" we should have different materials like
    // 'matte' and 'glass', 'metal' etc here, but for now, let's just
    // ignore the type and create a single one thta contains all
    // fields....
    return std::make_shared<Material>(dg);
  }

  void Material::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back({"material.baseColor", OWL_FLOAT3, base+OWL_OFFSETOF(DD,baseColor)});
    vars.push_back({"material.alphaTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,alphaTexture)});
    vars.push_back({"material.colorTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,colorTexture)});
    vars.push_back({"material.transmission", OWL_FLOAT, base+OWL_OFFSETOF(DD,transmission)});
    vars.push_back({"material.roughness", OWL_FLOAT, base+OWL_OFFSETOF(DD,roughness)});
    vars.push_back({"material.metallic", OWL_FLOAT, base+OWL_OFFSETOF(DD,metallic)});
    vars.push_back({"material.ior", OWL_FLOAT, base+OWL_OFFSETOF(DD,ior)});
  }

  void Material::set(OWLGeom geom) const
  {
    owlGeomSet3f(geom,"material.baseColor",
                 baseColor.x,
                 baseColor.y,
                 baseColor.z);
    owlGeomSet1f(geom,"material.transmission",transmission);
    owlGeomSet1f(geom,"material.roughness",roughness);
    owlGeomSet1f(geom,"material.metallic",metallic);
    owlGeomSet1f(geom,"material.ior",ior);
    owlGeomSetTexture(geom,"material.alphaTexture",
                      alphaTexture
                      ? alphaTexture->owlTex
                      : (OWLTexture)0
                      );
    owlGeomSetTexture(geom,"material.colorTexture",
                      colorTexture
                      ? colorTexture->owlTex
                      : (OWLTexture)0
                      );
  }

  void Material::commit()
  {
    DataGroupObject::commit();
  }
  
  bool Material::set1f(const std::string &member, const float &value)
  {
    if (DataGroupObject::set1f(member,value))
      return true;
    if (member == "transmission") {
      this->transmission = value;
      return true;
    }
    if (member == "ior") {
      this->ior = value;
      return true;
    }
    if (member == "metallic") {
      this->metallic = value;
      return true;
    }
    return false;
  }
  
  bool Material::set3f(const std::string &member, const vec3f &value)
  {
    if (DataGroupObject::set3f(member,value))
      return true;
    if (member == "baseColor") {
      this->baseColor = value;
      return true;
    }
    return false;
  }
  
  bool Material::setObject(const std::string &member, const Object::SP &value)
  {
    if (DataGroupObject::setObject(member,value))
      return true;
    if (member == "colorTexture") {
      this->colorTexture = value?value->as<Texture>():0;
      return true;
    }
    if (member == "alphaTexture") {
      this->alphaTexture = value?value->as<Texture>():0;
      return true;
    }
    return false;
  }
}
