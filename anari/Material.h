// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "Sampler.h"

namespace barney_device {

struct Material : public Object
{
  Material(BarneyGlobalState *s);
  ~Material() override;

  static Material *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  void markCommitted() override;

  virtual BNMaterial makeBarneyMaterial(BNModel model, int slot) const = 0;

};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Matte : public Material
{
  Matte(BarneyGlobalState *s);
  void commit() override;

  BNMaterial makeBarneyMaterial(BNModel model, int slot) const override;

 private:
  math::float4 m_color{1.f, 1.f, 1.f, 1.f};
  helium::IntrusivePtr<Sampler> m_colorSampler;
};

struct PhysicallyBased : public Material
{
  PhysicallyBased(BarneyGlobalState *s);
  void commit() override;

  BNMaterial makeBarneyMaterial(BNModel model, int slot) const override;

 private:
  struct {
    math::float4 value{1.f, 1.f, 1.f, 1.f};
    /*TODO: samplers, attributes, etc.*/
  } m_baseColor;

  struct {
    math::float3 value{1.f, 1.f, 1.f};
    /*TODO: samplers, attributes, etc.*/
  } m_emissive, m_specularColor;

  struct {
    float value{1.f};
    /*TODO: samplers, attributes, etc.*/
  } m_opacity, m_metallic, m_roughness, m_specular, m_transmission;

  float m_ior{1.5f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Material *, ANARI_MATERIAL);
