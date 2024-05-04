// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
#include "common.h"
// CUDA
#include <vector_functions.h>
#include <iostream>

namespace barney_device {

Material::Material(BarneyGlobalState *s) : Object(ANARI_MATERIAL, s) {}

Material::~Material()
{
  cleanup();
}

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

BNMaterial Material::getBarneyMaterial(BNModel model, int slot)
{
  if (!isModelTracked(model, slot)) {
    cleanup();
    trackModel(model, slot);
    m_bnMat = bnMaterialCreate(model, slot, bnSubtype());
    setBarneyParameters();
  }

  return m_bnMat;
}

void Material::cleanup()
{
  if (m_bnMat)
    bnRelease(m_bnMat);
  m_bnMat = nullptr;
}

// Subtypes ///////////////////////////////////////////////////////////////////

// Matte //

Matte::Matte(BarneyGlobalState *s) : Material(s), m_colorSampler(this) {}

void Matte::commit()
{
  Object::commit();

  if (m_colorSampler)
    m_colorSampler->removeCommitObserver(this);

  m_color = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("color", ANARI_FLOAT32_VEC3, &m_color);
  getParam("color", ANARI_FLOAT32_VEC4, &m_color);

  m_colorAttribute = getParamString("color","");
  m_colorSampler = getParamObject<Sampler>("color");

  setBarneyParameters();
}

bool Matte::isValid() const
{
  return !m_colorSampler || m_colorSampler->isValid();
}

const char *Matte::bnSubtype() const
{
#if 0 // barney 'matte' material is WIP
  return "matte";
#else
  return "physicallyBased";
#endif
}

void Matte::setBarneyParameters()
{
  if (!m_bnMat)
    return;
  bnSet3f(m_bnMat, "baseColor", m_color.x, m_color.y, m_color.z);
  if (m_colorSampler)
    m_colorSampler->setBarneyParameters(trackedModel(), m_bnMat, trackedSlot());
  if (!m_colorAttribute.empty())
    bnSetString(m_bnMat, "baseColor", m_colorAttribute.c_str());
  bnCommit(m_bnMat);
}

// PhysicallyBased //

PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s) {}

void PhysicallyBased::commit()
{
  Object::commit();

  m_baseColor.value = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_baseColor.value);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_baseColor.value);

  m_emissive.value = math::float3(0.f, 0.f, 0.f);
  getParam("emissive", ANARI_FLOAT32_VEC3, &m_emissive.value);

  m_specularColor.value = math::float3(1.f, 1.f, 1.f);
  getParam("specularColor", ANARI_FLOAT32_VEC3, &m_specularColor.value);

  m_opacity.value = 1.f;
  getParam("opacity", ANARI_FLOAT32, &m_opacity.value);

  m_metallic.value = 1.f;
  getParam("metallic", ANARI_FLOAT32, &m_metallic.value);
  m_metallic.stringValue = getParamString("metallic", "");

  // std::cout << "found metallic attribute " << metallicAttribute << std::endl;

  m_roughness.value = 1.f;
  getParam("roughness", ANARI_FLOAT32, &m_roughness.value);
  m_roughness.stringValue = getParamString("roughness", "");

  m_specular.value = 0.f;
  getParam("specular", ANARI_FLOAT32, &m_specular.value);

  m_transmission.value = 0.f;
  getParam("transmission", ANARI_FLOAT32, &m_transmission.value);

  m_ior = 1.5f;
  getParam("ior", ANARI_FLOAT32, &m_ior);

  setBarneyParameters();
}

const char *PhysicallyBased::bnSubtype() const
{
  return "physicallyBased";
}

void PhysicallyBased::setBarneyParameters()
{
  if (!m_bnMat)
    return;

  bnSet3f(m_bnMat,
      "baseColor",
      m_baseColor.value.x,
      m_baseColor.value.y,
      m_baseColor.value.z);

  bnSet3f(m_bnMat,
      "emissive",
      m_emissive.value.x,
      m_emissive.value.y,
      m_emissive.value.z);

  bnSet3f(m_bnMat,
      "specularColor",
      m_specularColor.value.x,
      m_specularColor.value.y,
      m_specularColor.value.z);

  bnSet1f(m_bnMat, "opacity", m_opacity.value);
  if (m_metallic.stringValue.empty())
    bnSet1f(m_bnMat, "metallic", m_metallic.value);
  else
    bnSetString(m_bnMat, "metallic", m_metallic.stringValue.c_str());
  if (m_roughness.stringValue.empty())
    bnSet1f(m_bnMat, "roughness", m_roughness.value);
  else
    bnSetString(m_bnMat, "roughness", m_roughness.stringValue.c_str());
  bnSet1f(m_bnMat, "specular", m_specular.value);
  bnSet1f(m_bnMat, "transmission", m_transmission.value);
  bnSet1f(m_bnMat, "ior", m_ior);

  bnCommit(m_bnMat);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
