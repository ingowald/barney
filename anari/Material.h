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

 protected:
  virtual const char *bnSubtype() const = 0;
  virtual void setBarneyParameters() = 0;
  void cleanup();
  BNMaterial m_bnMat{nullptr};
};

template <typename SCALAR_T>
struct MaterialParameter
{
  SCALAR_T value;
  std::string attribute;
  helium::IntrusivePtr<Sampler> sampler;
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
  MaterialParameter<math::float4> m_color;
  MaterialParameter<float> m_opacity;
};

struct PhysicallyBased : public Material
{
  PhysicallyBased(BarneyGlobalState *s);
  void commit() override;

  const char *bnSubtype() const override;
  void setBarneyParameters() override;

 private:
<<<<<<< Updated upstream
  MaterialParameter<math::float4> m_baseColor;
  MaterialParameter<math::float3> m_emissive, m_specularColor;
  MaterialParameter<float> m_opacity, m_metallic, m_roughness, m_specular,
      m_transmission;
=======
  struct
  {
    math::float4 value{1.f, 1.f, 1.f, 1.f};
    /*TODO: samplers, attributes, etc.*/
    std::string  stringValue;
    helium::CommitObserverPtr<Sampler> sampler{nullptr};
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
>>>>>>> Stashed changes

  float m_ior{1.5f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Material *, ANARI_MATERIAL);
