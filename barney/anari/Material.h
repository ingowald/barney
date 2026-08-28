// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari/Object.h"
#include "anari/Sampler.h"

namespace BARNEY_NS {
  namespace anari {

    struct Material : public Object
    {
      Material(BarneyGlobalState *s);
      ~Material() override;

      static Material *createInstance(std::string_view subtype,
                                      BarneyGlobalState *s);

      void finalize() override;

      BNMaterial getBarneyMaterial();

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
      description: control cut-out transparency, possible values: opaque, blend,
      mask

      <param>
      name: "alphaCutoff"
      type: FLOAT32
      default: 0.5
      description: threshold when alphaMode is mask
    */
    struct Matte : public Material
    {
      Matte(BarneyGlobalState *s);
      void commitParameters() override;
      bool isValid() const override;

      const char *bnSubtype() const override;
      void setBarneyParameters() override;

    private:
      MaterialParameter<math::float4> m_color;
      MaterialParameter<float>        m_opacity;
      std::string m_alphaMode{"opaque"};
      float m_alphaCutoff{0.5f};
    };

    // ==================================================================
    /*! ANARI "physiscallyBased" material

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
      void commitParameters() override;

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
      MaterialParameter<float> m_occlusion;
      MaterialParameter<float> m_clearcoat;
      MaterialParameter<float> m_clearcoatRoughness;
      MaterialParameter<math::float3> m_attenuationColor;
      MaterialParameter<float> m_thickness;
      MaterialParameter<float> m_attenuationDistance;
      MaterialParameter<math::float3> m_sheenColor;
      MaterialParameter<float> m_sheenRoughness;
      MaterialParameter<float> m_iridescence;
      MaterialParameter<float> m_iridescenceThickness;
      // Sampler-only params (no scalar default that maps cleanly).
      MaterialParameter<math::float4> m_normal;
      MaterialParameter<math::float4> m_clearcoatNormal;

      float m_ior{1.5f};
      float m_iridescenceIor{1.3f};
      std::string m_alphaMode{"opaque"};
      float m_alphaCutoff{0.5f};
    };

    // ==================================================================
    /*! ANARI "nvisii" material - Disney "Principled" BSDF (NVIDIA
        NVisii port). Exposes the full Disney parameter set, mapped
        1:1 to the barney core "nvisii" material subtype.

      <param>
      name: "baseColor"
      type: FLOAT32_VEC3 / SAMPLER / STRING
      default: (0.8, 0.8, 0.8)
      description: diffuse base color

      <param>
      name: "subsurfaceColor"
      type: FLOAT32_VEC3 / SAMPLER / STRING
      default: (0.8, 0.8, 0.8)
      description: subsurface scattering color

      <param>
      name: "metallic"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: metallic mask 0..1

      <param>
      name: "specular"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.5
      description: specular lobe strength

      <param>
      name: "roughness"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.5
      description: surface roughness 0..1

      <param>
      name: "specularTint"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: tint specular toward base color

      <param>
      name: "anisotropy"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: specular anisotropy -1..1

      <param>
      name: "sheen"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: sheen lobe strength

      <param>
      name: "sheenTint"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.5
      description: tint sheen toward base color

      <param>
      name: "clearcoat"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: clearcoat lobe strength

      <param>
      name: "clearcoatGloss"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.9991
      description: clearcoat gloss 0..1

      <param>
      name: "ior"
      type: FLOAT32 / SAMPLER / STRING
      default: 1.45
      description: index of refraction

      <param>
      name: "specularTransmission"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: specular transmission 0..1

      <param>
      name: "transmissionRoughness"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.04
      description: transmission lobe roughness

      <param>
      name: "flatness"
      type: FLOAT32 / SAMPLER / STRING
      default: 0.0
      description: mix of diffuse and subsurface

      <param>
      name: "opacity"
      type: FLOAT32 / SAMPLER / STRING
      default: 1.0
      description: cut-out opacity / alpha
    */
    struct NVisii : public Material
    {
      NVisii(BarneyGlobalState *s);
      void commitParameters() override;

      const char *bnSubtype() const override;
      void setBarneyParameters() override;

    private:
      MaterialParameter<math::float4> m_baseColor;
      MaterialParameter<math::float4> m_subsurfaceColor;
      MaterialParameter<float> m_metallic;
      MaterialParameter<float> m_specular;
      MaterialParameter<float> m_roughness;
      MaterialParameter<float> m_specularTint;
      MaterialParameter<float> m_anisotropy;
      MaterialParameter<float> m_sheen;
      MaterialParameter<float> m_sheenTint;
      MaterialParameter<float> m_clearcoat;
      MaterialParameter<float> m_clearcoatGloss;
      MaterialParameter<float> m_ior;
      MaterialParameter<float> m_specularTransmission;
      MaterialParameter<float> m_transmissionRoughness;
      MaterialParameter<float> m_flatness;
      MaterialParameter<float> m_opacity;
    };

    // ==================================================================
    /*! ANARI "glass" material - smooth dielectric (specular
        reflect/refract), mapped to the barney core "glass" material
        subtype. Barney-specific; not a KHR standard subtype.

      <param>
      name: "ior"
      type: FLOAT32
      default: 1.45
      description: index of refraction

      <param>
      name: "attenuationColor"
      type: FLOAT32_VEC3
      default: (1, 1, 1)
      description: Beer-Lambert transmission color
    */
    struct Glass : public Material
    {
      Glass(BarneyGlobalState *s);
      void commitParameters() override;

      const char *bnSubtype() const override;
      void setBarneyParameters() override;

    private:
      MaterialParameter<float> m_ior;
      MaterialParameter<math::float3> m_attenuationColor;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Material *, ANARI_MATERIAL);
