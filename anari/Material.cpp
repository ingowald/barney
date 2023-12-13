// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"

namespace barney_device {

Material::Material(BarneyGlobalState *s) : Object(ANARI_MATERIAL, s)
{
  s->objectCounts.materials++;
}

Material::~Material()
{
  deviceState()->objectCounts.materials--;
}

Material *Material::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
#if 0
  else if (subtype == "physicallyBased")
    return new PBM(s);
#endif
  else
    return (Material *)new UnknownObject(ANARI_MATERIAL, s);
}

void Material::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

const BNMaterial *Material::barneyMaterial() const
{
  return &m_bnMaterial;
}

// Subtypes ///////////////////////////////////////////////////////////////////

Matte::Matte(BarneyGlobalState *s) : Material(s) {}

void Matte::commit()
{
  Object::commit();

  m_bnMaterial.baseColor = make_float3(0.8f, 0.8f, 0.8f);
  m_bnMaterial.ior = 1.5f;
  m_bnMaterial.alphaTextureID = -1;
  m_bnMaterial.colorTextureID = -1;


  float4 color = make_float4(0.f, 0.f, 0.f, 1.f);
  if (getParam("color", ANARI_FLOAT32_VEC4, &color))
    std::memcpy(&m_bnMaterial.baseColor, &color, sizeof(float3));
  m_bnMaterial.baseColor = getParam<float3>("color", m_bnMaterial.baseColor);

  m_bnMaterial.transparency = getParam<float>("opacity", color.w);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
