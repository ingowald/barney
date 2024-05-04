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

  BNMaterial getBarneyMaterial(BNModel model, int slot);

  virtual const char *bnSubtype() const = 0;
  virtual void setBarneyParameters() = 0;

 protected:
  void cleanup();
  BNMaterial m_bnMat{nullptr};
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Matte : public Material
{
  Matte(BarneyGlobalState *s);
  void commit() override;
  bool isValid() const override;

  const char *bnSubtype() const override;
  void setBarneyParameters() override;

 private:
  math::float4 m_color{1.f, 1.f, 1.f, 1.f};
  std::string  m_colorAttribute;
  helium::CommitObserverPtr<Sampler> m_colorSampler{nullptr};
};

struct PhysicallyBased : public Material
{
  PhysicallyBased(BarneyGlobalState *s);
  void commit() override;

  const char *bnSubtype() const override;
  void setBarneyParameters() override;

 private:
  struct
  {
    math::float4 value{1.f, 1.f, 1.f, 1.f};
    /*TODO: samplers, attributes, etc.*/
    std::string  stringValue;
  } m_baseColor;

  struct
  {
    math::float3 value{1.f, 1.f, 1.f};
    /*TODO: samplers, attributes, etc.*/
  } m_emissive, m_specularColor;

  struct
  {
    float value{1.f};
    std::string stringValue;
    /*TODO: samplers, attributes, etc.*/
  } m_opacity, m_metallic, m_roughness, m_specular, m_transmission;

  float m_ior{1.5f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Material *, ANARI_MATERIAL);
