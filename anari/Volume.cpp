// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Volume.h"
// std
#if BARNEY_USE_MULTI_SCATTERING
#include <algorithm>
#else
#include <numeric>
#endif

namespace barney_device {

#if BARNEY_USE_MULTI_SCATTERING
  namespace {
    inline float luminance(const math::float3 &c)
    {
      return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    }
  }
#endif

  Volume::Volume(BarneyGlobalState *s) : Object(ANARI_VOLUME, s) {}

  Volume::~Volume()
  {
    cleanup();
  }

  Volume *Volume::createInstance(std::string_view subtype, BarneyGlobalState *s)
  {
    if (subtype == "transferFunction1D")
      return new TransferFunction1D(s);
#if BARNEY_USE_MULTI_SCATTERING
    if (subtype == "principled_volume")
      return new PrincipledVolume(s);
#endif
    return (Volume *)new UnknownObject(ANARI_VOLUME, subtype, s);
  }

  void Volume::markFinalized()
  {
    deviceState()->markStructuralSceneChanged();
    Object::markFinalized();
  }

  BNVolume Volume::getBarneyVolume()
  {
    if (!isValid())
      return {};
    if (!m_bnVolume) {
      m_bnVolume = createBarneyVolume();
      setBarneyParameters();
    }
    return m_bnVolume;
  }

  void Volume::cleanup()
  {
    if (m_bnVolume) {
      bnRelease(m_bnVolume);
      m_bnVolume = nullptr;
    }
  }

  void Volume::commitParameters()
  {
    m_id = getParam<uint32_t>("id", ~0u);
    m_visible = getParam<bool>("visible", true);
  }

  bool Volume::isVisible() const
  {
    return m_visible;
  }

#if !BARNEY_USE_MULTI_SCATTERING

  // TransferFunction1D (main) //////////////////////////////////////////////////

  TransferFunction1D::TransferFunction1D(BarneyGlobalState *s)
    : Volume(s), m_field(this), m_colorData(this), m_opacityData(this)
  {}

  bool TransferFunction1D::isValid() const
  {
    return m_field && m_field->isValid();
  }

  void TransferFunction1D::commitParameters()
  {
    Volume::commitParameters();
    m_field = getParamObject<SpatialField>("value");
    m_valueRange = getParam<box1>("valueRange", box1{0.f, 1.f});
    m_colorData = getParamObject<helium::Array1D>("color");
    m_uniformColor = math::float4(1.f);
    getParam("color", ANARI_FLOAT32_VEC3, &m_uniformColor);
    getParam("color", ANARI_FLOAT32_VEC4, &m_uniformColor);
    m_opacityData = getParamObject<helium::Array1D>("opacity");
    m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
    m_unitDistance = getParam<float>("unitDistance", 1.f);

    invalidateBarneyVolumeIfFieldChanged();
  }

  void TransferFunction1D::finalize()
  {
    if (!m_field) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "no spatial field provided to transferFunction1D volume");
      return;
    }

    if (m_unitDistance <= 0.f) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "invalid density scale calculated from 'unitDistance',"
                    " setting to 1");
      m_densityScale = 1.f;
    } else
      m_densityScale = 1.f / m_unitDistance;

    m_bounds = m_field->bounds();

    size_t numColorChannels{4};
    if (m_colorData) { // TODO: more types
      if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
        numColorChannels = 3;
    }

    float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
    float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

    size_t numColors = m_colorData ? m_colorData->size() : 1;
    size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
    size_t tfSize = std::max(numColors, numOpacities);

    m_rgbaMap.resize(tfSize);
    for (size_t i=0; i<tfSize; ++i) {
      float colorPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numColors-1) : 0.f;
      float colorFrac = colorPos-floorf(colorPos);

      math::float4 color0(m_uniformColor.xyz(), m_uniformOpacity);
      math::float4 color1(m_uniformColor.xyz(), m_uniformOpacity);
      if (colorData) {
        if (numColorChannels == 3) {
          math::float3 *colors = (math::float3 *)colorData;
          color0 = math::float4(colors[int(floorf(colorPos))], m_uniformOpacity);
          color1 = math::float4(colors[int(ceilf(colorPos))], m_uniformOpacity);
        }
        else if (numColorChannels == 4) {
          math::float4 *colors = (math::float4 *)colorData;
          color0 = colors[int(floorf(colorPos))];
          color1 = colors[int(ceilf(colorPos))];
        }
      }

      math::float4 color = (1.f-colorFrac)*color0 + colorFrac*color1;

      if (opacityData) {
        float alphaPos = tfSize > 1 ? (float(i)/(tfSize-1))*(numOpacities-1) : 0.f;
        float alphaFrac = alphaPos-floorf(alphaPos);

        float alpha0 = opacityData[int(floorf(alphaPos))];
        float alpha1 = opacityData[int(ceilf(alphaPos))];

        color.w *= (1.f-alphaFrac)*alpha0 + alphaFrac*alpha1;
      }

      m_rgbaMap[i] = color;
    }

    setBarneyParameters();
  }

  BNVolume TransferFunction1D::createBarneyVolume()
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    BNVolume volume = m_field
      ? bnVolumeCreate(context, slot, m_field->getBarneyScalarField())
      : BNVolume{};

    if (volume)
      m_boundField = m_field.get();

    return volume;
  }

  box3 TransferFunction1D::bounds() const
  {
    return m_bounds;
  }

  void TransferFunction1D::setBarneyParameters()
  {
    if (!isValid() || !m_bnVolume)
      return;
    BNVolume vol = getBarneyVolume();
    bnSet1i(vol,"userID",m_id);
    bnVolumeSetXF(vol,
                  (bn_float2 &)m_valueRange,
                  (const bn_float4 *)m_rgbaMap.data(),
                  (int)m_rgbaMap.size(),
                  m_densityScale);
    bnCommit(vol);
  }

  void TransferFunction1D::invalidateBarneyVolumeIfFieldChanged()
  {
    const SpatialField *newField = m_field.get();
    if (m_boundField == newField)
      return;

    cleanup();
    m_boundField = nullptr;
  }

#else

  // FieldMappedVolume //////////////////////////////////////////////////////////

  FieldMappedVolume::FieldMappedVolume(BarneyGlobalState *s)
    : Volume(s), m_field(this)
  {}

  bool FieldMappedVolume::isValid() const
  {
    return m_field && m_field->isValid();
  }

  void FieldMappedVolume::commitParameters()
  {
    Volume::commitParameters();
    m_field = getParamObject<SpatialField>("value");
    m_valueRange = getParam<box1>("valueRange", box1{0.f, 1.f});
    m_unitDistance = getParam<float>("unitDistance", 1.f);
    m_anisotropy = getParam<float>("anisotropy", 0.6f);
    m_scatteringAlbedo = getParam<float>("scatteringAlbedo", 0.9f);
    invalidateBarneyVolumeIfFieldChanged();
  }

  void FieldMappedVolume::finalizeFieldMappedVolume()
  {
    if (!m_field) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "no spatial field provided to volume");
      return;
    }

    if (m_unitDistance <= 0.f) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "invalid density scale calculated from 'unitDistance',"
                    " setting to 1");
      m_densityScale = 1.f;
    } else
      m_densityScale = 1.f / m_unitDistance;

    m_bounds = m_field->bounds();
    setBarneyParameters();
  }

  BNVolume FieldMappedVolume::createBarneyVolume()
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;

    BNVolume volume = m_field
      ? bnVolumeCreate(context, slot, m_field->getBarneyScalarField())
      : BNVolume{};

    if (volume)
      m_boundField = m_field.get();

    return volume;
  }

  box3 FieldMappedVolume::bounds() const
  {
    return m_bounds;
  }

  void FieldMappedVolume::setBarneyParameters()
  {
    if (!isValid() || !m_bnVolume)
      return;
    BNVolume vol = getBarneyVolume();
    bnSet1i(vol, "userID", m_id);
    bnSet1f(vol, "anisotropy", m_anisotropy);
    bnSet1f(vol, "scatteringAlbedo", m_scatteringAlbedo);
    bnSet1i(vol, "principledVolume", 0);
    bnVolumeSetXF(vol,
                  (bn_float2 &)m_valueRange,
                  (const bn_float4 *)m_rgbaMap.data(),
                  (int)m_rgbaMap.size(),
                  m_densityScale);
    bnCommit(vol);
  }

  void FieldMappedVolume::invalidateBarneyVolumeIfFieldChanged()
  {
    const SpatialField *newField = m_field.get();
    if (m_boundField == newField)
      return;

    cleanup();
    m_boundField = nullptr;
  }

  // TransferFunction1D ///////////////////////////////////////////////////////////

  TransferFunction1D::TransferFunction1D(BarneyGlobalState *s)
    : FieldMappedVolume(s), m_colorData(this), m_opacityData(this)
  {}

  void TransferFunction1D::commitParameters()
  {
    FieldMappedVolume::commitParameters();
    m_colorData = getParamObject<helium::Array1D>("color");
    math::float4 uniformColor(1.f);
    getParam("color", ANARI_FLOAT32_VEC3, &uniformColor);
    getParam("color", ANARI_FLOAT32_VEC4, &uniformColor);
    m_uniformColor = uniformColor;
    m_opacityData = getParamObject<helium::Array1D>("opacity");
    m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
  }

  void TransferFunction1D::finalize()
  {
    if (!m_field) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "no spatial field provided to transferFunction1D volume");
      return;
    }

    size_t numColorChannels{4};
    if (m_colorData) {
      if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
        numColorChannels = 3;
    }

    float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
    float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

    size_t numColors = m_colorData ? m_colorData->size() : 1;
    size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
    size_t tfSize = std::max(numColors, numOpacities);

    m_rgbaMap.resize(tfSize);
    for (size_t i = 0; i < tfSize; ++i) {
      float colorPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numColors - 1) : 0.f;
      float colorFrac = colorPos - floorf(colorPos);

      math::float4 color0(m_uniformColor.xyz(), m_uniformOpacity);
      math::float4 color1(m_uniformColor.xyz(), m_uniformOpacity);
      if (colorData) {
        if (numColorChannels == 3) {
          math::float3 *colors = (math::float3 *)colorData;
          color0 = math::float4(colors[int(floorf(colorPos))], m_uniformOpacity);
          color1 = math::float4(colors[int(ceilf(colorPos))], m_uniformOpacity);
        } else if (numColorChannels == 4) {
          math::float4 *colors = (math::float4 *)colorData;
          color0 = colors[int(floorf(colorPos))];
          color1 = colors[int(ceilf(colorPos))];
        }
      }

      math::float4 color = (1.f - colorFrac) * color0 + colorFrac * color1;

      if (opacityData) {
        float alphaPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numOpacities - 1) : 0.f;
        float alphaFrac = alphaPos - floorf(alphaPos);

        float alpha0 = opacityData[int(floorf(alphaPos))];
        float alpha1 = opacityData[int(ceilf(alphaPos))];

        color.w *= (1.f - alphaFrac) * alpha0 + alphaFrac * alpha1;
      }

      m_rgbaMap[i] = color;
    }

    FieldMappedVolume::finalizeFieldMappedVolume();
  }

  // PrincipledVolume ///////////////////////////////////////////////////////////

  PrincipledVolume::PrincipledVolume(BarneyGlobalState *s)
    : FieldMappedVolume(s),
      m_colorData(this),
      m_opacityData(this)
  {}

  void PrincipledVolume::rebuildRGBAMapFromTransferFunction()
  {
    size_t numColorChannels{4};
    if (m_colorData) {
      if (m_colorData->elementType() == ANARI_FLOAT32_VEC3)
        numColorChannels = 3;
    }

    float *colorData = m_colorData ? (float *)m_colorData->data() : nullptr;
    float *opacityData = m_opacityData ? (float *)m_opacityData->data() : nullptr;

    size_t numColors = m_colorData ? m_colorData->size() : 1;
    size_t numOpacities = m_opacityData ? m_opacityData->size() : 1;
    size_t tfSize = std::max(numColors, numOpacities);

    m_rgbaMap.resize(tfSize);
    for (size_t i = 0; i < tfSize; ++i) {
      float colorPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numColors - 1) : 0.f;
      float colorFrac = colorPos - floorf(colorPos);

      math::float4 color0(m_uniformColor.xyz(), m_uniformOpacity);
      math::float4 color1(m_uniformColor.xyz(), m_uniformOpacity);
      if (colorData) {
        if (numColorChannels == 3) {
          math::float3 *colors = (math::float3 *)colorData;
          color0 = math::float4(colors[int(floorf(colorPos))], m_uniformOpacity);
          color1 = math::float4(colors[int(ceilf(colorPos))], m_uniformOpacity);
        } else if (numColorChannels == 4) {
          math::float4 *colors = (math::float4 *)colorData;
          color0 = colors[int(floorf(colorPos))];
          color1 = colors[int(ceilf(colorPos))];
        }
      }

      math::float4 color = (1.f - colorFrac) * color0 + colorFrac * color1;

      if (opacityData) {
        float alphaPos = tfSize > 1 ? (float(i) / (tfSize - 1)) * (numOpacities - 1) : 0.f;
        float alphaFrac = alphaPos - floorf(alphaPos);

        float alpha0 = opacityData[int(floorf(alphaPos))];
        float alpha1 = opacityData[int(ceilf(alphaPos))];

        color.w *= (1.f - alphaFrac) * alpha0 + alphaFrac * alpha1;
      }

      m_rgbaMap[i] = color;
    }
  }

  void PrincipledVolume::commitParameters()
  {
    FieldMappedVolume::commitParameters();
    m_density = getParam<float>("density", 1.f);
    m_color = getParam<math::float3>("scatterColor",
                                     getParam<math::float3>("color",
                                                            math::float3(0.8f)));
    m_absorptionColor = getParam<math::float3>("absorptionColor", math::float3(0.f));
    m_densityThreshold = getParam<float>("densityThreshold", 0.f);
    m_emissionStrength = getParam<float>("emissionStrength", 0.f);
    m_emissionColor = getParam<math::float3>("emissionColor", math::float3(1.f));
    m_blackbodyIntensity = getParam<float>("blackbodyIntensity", 0.f);
    m_blackbodyTint = getParam<math::float3>("blackbodyTint", math::float3(1.f));
    m_temperature = getParam<float>("temperature", 0.f);

    m_colorData = getParamObject<helium::Array1D>("color");
    math::float4 uniformColor(1.f);
    getParam("color", ANARI_FLOAT32_VEC3, &uniformColor);
    getParam("color", ANARI_FLOAT32_VEC4, &uniformColor);
    m_uniformColor = uniformColor;
    m_opacityData = getParamObject<helium::Array1D>("opacity");
    m_uniformOpacity = getParam<float>("opacity", 1.f) * m_uniformColor.w;
    rebuildRGBAMapFromTransferFunction();

    if (!hasParam("scatteringAlbedo")) {
      float absorb = luminance(m_absorptionColor);
      m_scatteringAlbedo = absorb < 0.f ? 0.f : (absorb > 1.f ? 1.f : 1.f - absorb);
    }

    if (m_unitDistance <= 0.f)
      m_densityScale = 1.f;
    else
      m_densityScale = 1.f / m_unitDistance;

    if (m_bnVolume)
      setBarneyParameters();
  }

  void PrincipledVolume::setBarneyParameters()
  {
    if (!isValid())
      return;

    if (m_unitDistance <= 0.f)
      m_densityScale = 1.f;
    else
      m_densityScale = 1.f / m_unitDistance;

    if (!m_bnVolume)
      return;

    BNVolume vol = m_bnVolume;
    bnSet1i(vol, "userID", m_id);
    bnSet1f(vol, "anisotropy", m_anisotropy);
    bnSet1f(vol, "scatteringAlbedo", m_scatteringAlbedo);
    bnSet1i(vol, "principledVolume", 1);
    bnSet1f(vol, "principledDensity", m_density);
    bnSet3f(vol, "principledScatterColor", m_color.x, m_color.y, m_color.z);
    bnSet3f(vol, "principledAbsorptionColor",
            m_absorptionColor.x, m_absorptionColor.y, m_absorptionColor.z);
    bnSet1f(vol, "principledDensityThreshold", m_densityThreshold);
    bnSet1f(vol, "principledDensityScale", m_densityScale);
    bnSet2f(vol, "principledValueRange", m_valueRange.lower, m_valueRange.upper);
    bnSet1f(vol, "principledEmissionStrength", m_emissionStrength);
    bnSet3f(vol, "principledEmissionColor",
            m_emissionColor.x, m_emissionColor.y, m_emissionColor.z);
    bnSet1f(vol, "principledBlackbodyIntensity", m_blackbodyIntensity);
    bnSet3f(vol, "principledBlackbodyTint",
            m_blackbodyTint.x, m_blackbodyTint.y, m_blackbodyTint.z);
    bnSet1f(vol, "principledTemperature", m_temperature);

    if (!m_rgbaMap.empty()) {
      bnVolumeSetXF(vol,
                    (bn_float2 &)m_valueRange,
                    (const bn_float4 *)m_rgbaMap.data(),
                    (int)m_rgbaMap.size(),
                    m_densityScale);
    }
    bnCommit(vol);
  }

  void PrincipledVolume::finalize()
  {
    if (!m_field) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "no spatial field provided to principled_volume");
      return;
    }

    rebuildRGBAMapFromTransferFunction();

    if (m_unitDistance <= 0.f) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "invalid density scale calculated from 'unitDistance',"
                    " setting to 1");
      m_densityScale = 1.f;
    } else
      m_densityScale = 1.f / m_unitDistance;

    m_bounds = m_field->bounds();
    setBarneyParameters();
  }

#endif

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Volume *);
