// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
#include "common.h"
// CUDA
#include <vector_functions.h>
#include <iostream>

namespace barney_device {

// Helper functions ///////////////////////////////////////////////////////////

template <typename T>
inline MaterialParameter<T> getMaterialHelper(
    Object *o, const char *p, T defaultValue)
{
  MaterialParameter<T> retval;
  retval.value = o->getParam<T>(p, defaultValue);
  retval.attribute = o->getParamString(p, "");
  retval.sampler = o->getParamObject<Sampler>(p);
  return retval;
}

template <>
inline MaterialParameter<math::float4> getMaterialHelper(
    Object *o, const char *p, math::float4 defaultValue)
{
  MaterialParameter<math::float4> retval;
  retval.value = defaultValue;
  o->getParam(p, ANARI_FLOAT32_VEC3, &retval.value);
  o->getParam(p, ANARI_FLOAT32_VEC4, &retval.value);
  retval.attribute = o->getParamString(p, "");
  retval.sampler = o->getParamObject<Sampler>(p);
  return retval;
}

template <typename T>
inline void setBNMaterialUniform(BNMaterial m, const char *p, T v)
{
  throw std::runtime_error("unhandled setBNMaterialUniform type");
}

template <>
inline void setBNMaterialUniform(BNMaterial m, const char *p, float v)
{
  bnSet1f(m, p, v);
}

template <>
inline void setBNMaterialUniform(BNMaterial m, const char *p, math::float3 v)
{
  bnSet3f(m, p, v.x, v.y, v.z);
}

template <>
inline void setBNMaterialUniform(BNMaterial m, const char *p, math::float4 v)
{
  bnSet4f(m, p, v.x, v.y, v.z, v.w);
}

template <typename T>
inline void setBNMaterialHelper(BNMaterial m,
    const char *p,
    MaterialParameter<T> &mp,
    BNModel model,
    int slot)
{
  if (mp.sampler) {
    BNSampler s = mp.sampler->getBarneySampler(model, slot);
    // TODO: set sampler
  } else if (!mp.attribute.empty())
    bnSetString(m, "baseColor", mp.attribute.c_str());
  else
    setBNMaterialUniform(m, p, mp.value);
}

// Material definitions ///////////////////////////////////////////////////////

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

Matte::Matte(BarneyGlobalState *s) : Material(s)
{
  this->commit(); // init with defaults for scalar values
}

void Matte::commit()
{
  Object::commit();
  m_color = getMaterialHelper(this, "color", math::float4(0.8f, 0.8f, 0.8f, 1));
  m_opacity = getMaterialHelper(this, "opacity", 1.f);
  setBarneyParameters();
}

bool Matte::isValid() const
{
  return !m_color.sampler || m_color.sampler->isValid();
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
<<<<<<< Updated upstream

  BNModel model = trackedModel();
  int slot = trackedSlot();

  // NOTE: using Barney PBR material because matte wasn't (isn't?) finished
  setBNMaterialHelper(m_bnMat, "baseColor", m_color, model, slot);
  setBNMaterialHelper(m_bnMat, "opacity", m_opacity, model, slot);
  bnSet1f(m_bnMat, "metallic", 0.f);
  bnSet1f(m_bnMat, "roughness", 1.f);
  bnSet1f(m_bnMat, "specular", 0.f);
  bnSet1f(m_bnMat, "transmission", 0.f);
=======
  if (!m_colorAttribute.empty())
    bnSetString(m_bnMat, "baseColor", m_colorAttribute.c_str());
  else if (m_colorSampler)
    bnSetObject(m_bnMat, "baseColor",
                m_baseColor.sampler->getBarneySampler(trackedModel(), trackedSlot()));
  else
    bnSet3f(m_bnMat, "baseColor", m_color.x, m_color.y, m_color.z);
>>>>>>> Stashed changes
  bnCommit(m_bnMat);
}

// PhysicallyBased //

<<<<<<< Updated upstream
PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s)
{
  this->commit(); // init with defaults for scalar values
=======
PhysicallyBased::PhysicallyBased(BarneyGlobalState *s)
  : Material(s)
{
  m_baseColor.sampler = helium::CommitObserverPtr<Sampler>(this);
>>>>>>> Stashed changes
}

void PhysicallyBased::commit()
{
  Object::commit();
<<<<<<< Updated upstream
  m_baseColor = getMaterialHelper(this, "baseColor", math::float4(1, 1, 1, 1));
  m_emissive = getMaterialHelper(this, "emissive", math::float3(0, 0, 0));
  m_specularColor =
      getMaterialHelper(this, "specularColor", math::float3(1, 1, 1));
  m_opacity = getMaterialHelper(this, "opacity", 1.f);
  m_metallic = getMaterialHelper(this, "metallic", 1.f);
  m_roughness = getMaterialHelper(this, "roughness", 1.f);
  m_specular = getMaterialHelper(this, "specular", 0.f);
  m_transmission = getMaterialHelper(this, "transmission", 0.f);
  m_ior = getParam<float>("ior", 1.5f);
=======

  m_baseColor.value = math::float4(1.f, 1.f, 1.f, 1.f);
  getParam("baseColor", ANARI_FLOAT32_VEC3, &m_baseColor.value);
  getParam("baseColor", ANARI_FLOAT32_VEC4, &m_baseColor.value);
  m_baseColor.stringValue = getParamString("baseColor","");
  m_baseColor.sampler     = getParamObject<Sampler>("baseColor");

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

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
  BNModel model = trackedModel();
  int slot = trackedSlot();

  setBNMaterialHelper(m_bnMat, "baseColor", m_baseColor, model, slot);
  setBNMaterialHelper(m_bnMat, "emissive", m_emissive, model, slot);
  setBNMaterialHelper(m_bnMat, "specularColor", m_specularColor, model, slot);
  setBNMaterialHelper(m_bnMat, "opacity", m_opacity, model, slot);
  setBNMaterialHelper(m_bnMat, "metallic", m_metallic, model, slot);
  setBNMaterialHelper(m_bnMat, "roughness", m_roughness, model, slot);
  setBNMaterialHelper(m_bnMat, "specular", m_specular, model, slot);
  setBNMaterialHelper(m_bnMat, "transmission", m_transmission, model, slot);
  setBNMaterialHelper(m_bnMat, "opacity", m_opacity, model, slot);
=======
  if (m_baseColor.sampler)
    bnSetObject(m_bnMat, "baseColor",
                m_baseColor.sampler->getBarneySampler(trackedModel(), trackedSlot()));
  else if (!m_colorAttribute.empty())
    bnSetString(m_bnMat, "baseColor", m_colorAttribute.c_str());
  else
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
>>>>>>> Stashed changes
  bnSet1f(m_bnMat, "ior", m_ior);

  bnCommit(m_bnMat);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
