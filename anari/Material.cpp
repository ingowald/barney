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

// Subtypes ///////////////////////////////////////////////////////////////////

// Matte //

Matte::Matte(BarneyGlobalState *s) : Material(s) {}

void Matte::commit()
{
  Object::commit();

  m_color = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_color);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);
}

BNMaterial Matte::makeBarneyMaterial(BNModel model, int slot) const
{
  BNMaterial mat = bnMaterialCreate(model, slot, "velvet");
  bnSet3f(mat, "reflectance", m_color.x, m_color.y, m_color.z);
  return mat;
}

// PhysicallyBased //

PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s) {}

void PhysicallyBased::commit()
{
  Object::commit();

  m_baseColor.value = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_baseColor.value);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_baseColor.value);

  m_opacity.value = 1.f;
  getParam("opacity", ANARI_FLOAT32, &m_opacity.value);

  m_metallic.value = 1.f;
  getParam("metallic", ANARI_FLOAT32, &m_metallic.value);

  m_roughness.value = 1.f;
  getParam("roughness", ANARI_FLOAT32, &m_roughness.value);

  m_ior = 1.5f;
  getParam("ior", ANARI_FLOAT32, &m_ior);
}

BNMaterial PhysicallyBased::makeBarneyMaterial(BNModel model, int slot) const
{
  BNMaterial mat = bnMaterialCreate(model, slot, "velvet");
  bnSet3f(mat, "reflectance",
      m_baseColor.value.x, m_baseColor.value.y, m_baseColor.value.z);
  return mat;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
