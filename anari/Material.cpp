// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
#include "common.h"
// // CUDA
// #include <vector_functions.h>
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
                                MaterialParameter<T> &mp)
{
  if (mp.sampler) {
    BNSampler s = mp.sampler->getBarneySampler();
    bnSetObject(m, p, s);
  } else if (!mp.attribute.empty())
    bnSetString(m, p, mp.attribute.c_str());
  else {
    setBNMaterialUniform(m, p, mp.value);
  }
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

void Material::finalize()
{
  setBarneyParameters();
}

BNMaterial Material::getBarneyMaterial()
{
  int slot = deviceState()->slot;
  auto context = deviceState()->tether->context;

  if (!m_bnMat) 
    m_bnMat = bnMaterialCreate(context, slot, bnSubtype());
  setBarneyParameters();
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
  this->commitParameters(); // init with defaults for scalar values
}

void Matte::commitParameters()
{
  Object::commitParameters();
  m_color = getMaterialHelper(this, "color", math::float4(0.8f, 0.8f, 0.8f, 1));
  m_opacity = getMaterialHelper(this, "opacity", 1.f);
}

bool Matte::isValid() const
{
  return !m_color.sampler || m_color.sampler->isValid();
}

const char *Matte::bnSubtype() const
{
  return "AnariMatte";
}

void Matte::setBarneyParameters()
{
  if (!m_bnMat)
    return;

  setBNMaterialHelper(m_bnMat, "color", m_color);//, getContext());
  bnCommit(m_bnMat);
}

// PhysicallyBased //

PhysicallyBased::PhysicallyBased(BarneyGlobalState *s) : Material(s)
{
  this->commitParameters(); // init with defaults for scalar values
}

void PhysicallyBased::commitParameters()
{
  Object::commitParameters();
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
  m_opacity = getMaterialHelper(this, "opacity", 1.f);
}

const char *PhysicallyBased::bnSubtype() const
{
  return "physicallyBased";
}

void PhysicallyBased::setBarneyParameters()
{
  if (!m_bnMat)
    return;

  // auto context = deviceState()->tether->context;//getContext();
  // int slot = deviceState()->slot;

  setBNMaterialHelper(m_bnMat, "baseColor", m_baseColor);
  setBNMaterialHelper(m_bnMat, "emissive", m_emissive);
  setBNMaterialHelper(m_bnMat, "specularColor", m_specularColor);
  setBNMaterialHelper(m_bnMat, "metallic", m_metallic);
  setBNMaterialHelper(m_bnMat, "roughness", m_roughness);
  setBNMaterialHelper(m_bnMat, "specular", m_specular);
  setBNMaterialHelper(m_bnMat, "transmission", m_transmission);
  setBNMaterialHelper(m_bnMat, "opacity", m_opacity);

  bnSet1f(m_bnMat, "ior", m_ior);
  bnCommit(m_bnMat);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Material *);
