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

  BNMaterial getBarneyMaterial(BNContext context// , int slot
                               );

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

  // ==================================================================
  /*! ANARI "matte" material

    from anari spec:
    <param>
    name: "color"
    type: FLOAT32_VEC3 / SAMPLER / STRING
    default: (0.8, 0.8, 0.8)
    description: diffuse color

    <param>
    name: "opacity"
    type: FLOAT32 / SAMPLER / STRING
    default: 1.0             opacity

    <param>
    name: "alphaMode"
    type:  STRING
    default: "opaque"
    description: control cut-out transparency, possible values: opaque, blend, mask

    <param>
    name: "alphaCutoff"
    type: FLOAT32
    default: 0.5
    description: threshold when alphaMode is mask
  */
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

  // ==================================================================
  /*! ANARI "matte" material

    from anari spec:

    <param>
    baseColor
    FLOAT32_VEC3 / SAMPLER / STRING
    (1.0, 1.0, 1.0)

    <param>
    opacity
    FLOAT32 / SAMPLER / STRING
    1.0

    <param>
    metallic
    FLOAT32 / SAMPLER / STRING
    1.0

    <param>
    roughness
    FLOAT32 / SAMPLER / STRING
    1.0

    <param>
    normal
    SAMPLER

    <param>
    emissive
    FLOAT32_VEC3 / SAMPLER / STRING
    (0.0, 0.0, 0.0)

    <param>
    occlusion
    SAMPLER

    <param>
    alphaMode
    STRING
    opaque

    <param>
    alphaCutoff
    FLOAT32
    0.5

    <param>
    specular
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    specularColor
    FLOAT32_VEC3 / SAMPLER / STRING
    (1.0, 1.0, 1.0)

    <param>
    clearcoat
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    clearcoatRoughness
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    clearcoatNormal
    SAMPLER

    <param>
    transmission
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    ior
    FLOAT32
    1.5

    <param>
    thickness
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    attenuationDistance
    FLOAT32
    INF

    <param>
    attenuationColor
    FLOAT32_VEC3
    (1.0, 1.0, 1.0)

    <param>
    sheenColor
    FLOAT32_VEC3 / SAMPLER / STRING
    (0.0, 0.0, 0.0)

    <param>
    sheenRoughness
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    iridescence
    FLOAT32 / SAMPLER / STRING
    0.0

    <param>
    iridescenceIor
    FLOAT32
    1.3

    <param>
    iridescenceThickness
    FLOAT32 / SAMPLER / STRING
    0.0
  */
struct PhysicallyBased : public Material
{
  PhysicallyBased(BarneyGlobalState *s);
  void commit() override;

  const char *bnSubtype() const override;
  void setBarneyParameters() override;

 private:
  MaterialParameter<math::float4> m_baseColor;
  MaterialParameter<math::float3> m_emissive;
  MaterialParameter<math::float3> m_specularColor;
  MaterialParameter<float> m_opacity;
  MaterialParameter<float> m_metallic;
  MaterialParameter<float> m_roughness;
  MaterialParameter<float> m_specular;
  MaterialParameter<float> m_transmission;

  float m_ior{1.5f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Material *, ANARI_MATERIAL);
