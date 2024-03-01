// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
// CUDA
#include <vector_functions.h>

namespace barney_device {

Material::Material(BarneyGlobalState *s) : Object(ANARI_MATERIAL, s) {}

Material::~Material() = default;

Material *Material::createInstance(
    std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
  else if (subtype == "physicallyBased")
    return new PhysicallyBased(s);
  else
    return (Material *)new UnknownObject(ANARI_MATERIAL, s);
}

void Material::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

const BNMaterialHelper *Material::barneyMaterial() const
{
  return &m_bnMaterialData;
}

// Subtypes ///////////////////////////////////////////////////////////////////

// Matte //

Matte::Matte(BarneyGlobalState *s) : Material(s) {}

void Matte::commit()
{
  Object::commit();

  m_bnMaterialData.ior = 1.5f;
  m_bnMaterialData.alphaTexture = 0;
  m_bnMaterialData.colorTexture = 0;

  math::float4 color(0.8f, 0.8f, 0.8f, 1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &color);
  getParam("color", ANARI_FLOAT32_VEC4, &color);
  std::memcpy(&m_bnMaterialData.baseColor, &color, sizeof(float3));

  m_bnMaterialData.transmission = getParam<float>("opacity", color.w);
  m_bnMaterialData.ior = 1.f;
  m_bnMaterialData.metallic = 0.f;
  m_bnMaterialData.roughness = 0.f;
  printf("TODO: set material data....\n");
}

// PhysicallyBased //

PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s) {}

void PhysicallyBased::commit()
{
  Object::commit();

  m_bnMaterialData.alphaTexture = 0;
  m_bnMaterialData.colorTexture = 0;

  math::float4 color(0.8f, 0.8f, 0.8f, 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &color);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &color);
  std::memcpy(&m_bnMaterialData.baseColor, &color, sizeof(float3));

  m_bnMaterialData.transmission = getParam<float>("opacity", color.w);
  m_bnMaterialData.ior = 1.f;
  m_bnMaterialData.metallic = 0.f;
  m_bnMaterialData.roughness = 0.f;
  printf("TODO: set material data....\n");
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
