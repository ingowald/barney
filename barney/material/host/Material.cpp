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

#include "barney/material/host/Velvet.h"
#include "barney/material/host/Matte.h"
#include "barney/material/host/Metal.h"
#include "barney/material/host/Plastic.h"
#include "barney/material/host/MetallicPaint.h"
#include "barney/ModelSlot.h"

namespace barney {

  struct AnariPhysicalMaterial : public barney::Material {
    AnariPhysicalMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~AnariPhysicalMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "<AnariPhysicalMaterial>"; }

    void createDD(DD &dd, int deviceID) const override
    {
      std::cout << "baseColor: " << baseColor << '\n';
      std::cout << "emissive: " << emissive << '\n';
      std::cout << "specularColor: " << specularColor << '\n';
      std::cout << "opacity: " << metallic << '\n';
      std::cout << "metallic: " << metallic << '\n';
      std::cout << "roughness: " << roughness << '\n';
      std::cout << "transmission: " << transmission << '\n';
      std::cout << "ior: " << ior << '\n';
    }
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override {};
    bool set1f(const std::string &member, const float &value) override
    {
      if (Material::set1f(member,value)) return true;
      if (member == "opacity")
        { opacity = value; return true; }
      if (member == "metallic")
        { metallic = value; return true; }
      if (member == "roughness")
        { roughness = value; return true; }
      if (member == "specular")
        { specular = value; return true; }
      if (member == "transmmission")
        { transmission = value; return true; }
      if (member == "ior")
        { ior = value; return true; }
      return false;
    }
    bool set3f(const std::string &member, const vec3f &value) override
    {
      if (Material::set3f(member,value)) return true;
      if (member == "baseColor")
        { baseColor = value; return true; }
      return false;
    }
    /*! @} */
    // ------------------------------------------------------------------
    /* iw - i have NO CLUE what goes in here .... */
    vec3f baseColor { 1.f, 1.f, 1.f };
    vec3f emissive { 0.f, 0.f, 0.f };
    vec3f specularColor { 1.f, 1.f, 1.f };
    float opacity = 1.f;
    float metallic = 1.f;
    float roughness = 1.f;
    float specular = 0.f;
    float transmission = 0.f;
    float ior = 0.f;
  };

  /*! material according to "miniScene" default specification. will
    internally build a AnariPhyisical device data */
  struct MiniMaterial : public barney::Material {
    MiniMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~MiniMaterial() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "MiniMaterial"; }

    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    vec3f baseColor { .5f, .5f, .5f };
    vec3f emission  { 0.f };
    float transmission { 0.f };
    float roughness    { 0.f };
    float metallic     { 0.f };
    float ior          { 1.f };
    Texture::SP colorTexture;
    Texture::SP alphaTexture;
  };

  void MiniMaterial::commit()
  { /* we dont' yet stage/double-buffer params ... */}
  
  void MiniMaterial::createDD(DD &dd, int deviceID) const
  {
    dd.materialType = render::MINI;
    dd.mini.ior = ior;
    dd.mini.transmission = transmission;
    dd.mini.baseColor = baseColor;
    dd.mini.colorTexture
      = colorTexture
      ? owlTextureGetObject(colorTexture->owlTex,deviceID)
      : 0;
    dd.mini.alphaTexture
      = alphaTexture
      ? owlTextureGetObject(alphaTexture->owlTex,deviceID)
      : 0;
  }

  Material::SP Material::create(ModelSlot *dg, const std::string &type)
  {
    if (type == "velvet")
      return std::make_shared<VelvetMaterial>(dg);
    if (type == "matte")
      return std::make_shared<MatteMaterial>(dg); 
    if (type == "metal")
      return std::make_shared<MetalMaterial>(dg); 
    if (type == "plastic")
      return std::make_shared<PlasticMaterial>(dg);
    if (type == "metallic_paint")
      return std::make_shared<MetallicPaintMaterial>(dg);
    if (type == "velvet")
      return std::make_shared<VelvetMaterial>(dg);
    else if (type == "physicallyBased")
      return std::make_shared<AnariPhysicalMaterial>(dg);
    // iw - "eventually" we should have different materials like
    // 'matte' and 'glass', 'metal' etc here, but for now, let's just
    // ignore the type and create a single one thta contains all
    // fields....
    return std::make_shared<MiniMaterial>(dg);
  }

  void Material::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back({"material", OWL_USER_TYPE(Material::DD), base+0u});
    // vars.push_back({"material.baseColor", OWL_FLOAT3, base+OWL_OFFSETOF(DD,baseColor)});
    // vars.push_back({"material.alphaTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,alphaTexture)});
    // vars.push_back({"material.colorTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,colorTexture)});
    // vars.push_back({"material.transmission", OWL_FLOAT, base+OWL_OFFSETOF(DD,transmission)});
    // vars.push_back({"material.roughness", OWL_FLOAT, base+OWL_OFFSETOF(DD,roughness)});
    // vars.push_back({"material.metallic", OWL_FLOAT, base+OWL_OFFSETOF(DD,metallic)});
    // vars.push_back({"material.ior", OWL_FLOAT, base+OWL_OFFSETOF(DD,ior)});
  }

  void Material::setDeviceDataOn(OWLGeom geom) const
  {
    for (int deviceID=0;deviceID<owner->devGroup->size();deviceID++) {
      Material::DD dd;
      createDD(dd,deviceID);
      owlGeomSetRaw(geom,"material",&dd,deviceID);
    }
  }

  void Material::commit()
  {
    SlottedObject::commit();
  }

  bool MiniMaterial::set1f(const std::string &member, const float &value)
  {
    if (SlottedObject::set1f(member,value))
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
    if (member == "roughness") {
      this->roughness = value;
      return true;
    }
    return false;
  }
  
  bool MiniMaterial::set3f(const std::string &member, const vec3f &value)
  {
    if (SlottedObject::set3f(member,value))
      return true;
    if (member == "baseColor") {
      this->baseColor = value;
      return true;
    }
    if (member == "emission") {
      this->emission = value;
      return true;
    }
    return false;
  }
  
  bool MiniMaterial::setObject(const std::string &member, const Object::SP &value)
  {
    if (SlottedObject::setObject(member,value))
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
