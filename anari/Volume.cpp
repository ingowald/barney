// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Volume.h"
// std
#include <numeric>

namespace barney_device {

Volume::Volume(BarneyGlobalState *s) : Object(ANARI_VOLUME, s) {}

Volume::~Volume() = default;

Volume *Volume::createInstance(std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "transferFunction1D")
    return new TransferFunction1D(s);
  else
    return (Volume *)new UnknownObject(ANARI_VOLUME, s);
}

void Volume::markCommitted()
{
  deviceState()->markSceneChanged();
  Object::markCommitted();
}

BNVolume Volume::getBarneyVolume(BNModel model, int slot)
{
  if (!isValid())
    return {};
  if (!isModelTracked(model, slot)) {
    cleanup();
    trackModel(model, slot);
    m_bnVolume = createBarneyVolume(model, slot);
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

// Subtypes ///////////////////////////////////////////////////////////////////

TransferFunction1D::TransferFunction1D(BarneyGlobalState *s) : Volume(s) {}

bool TransferFunction1D::isValid() const
{
  return m_field && m_field->isValid() && m_colorData && m_opacityData;
}

void TransferFunction1D::commit()
{
  Volume::commit();

  // cleanup();

  m_field = getParamObject<SpatialField>("value");
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  m_bounds = m_field->bounds();

  m_valueRange = getParam<box1>("valueRange", box1{0.f, 1.f});

  m_colorData = getParamObject<helium::Array1D>("color");
  m_opacityData = getParamObject<helium::Array1D>("opacity");
  m_densityScale = getParam<float>("unitDistance", 1.f);

  if (!m_colorData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no color data provided to transferFunction1D volume");
    return;
  }

  if (!m_opacityData) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no opacity data provided to transfer function");
    return;
  }

  // extract combined RGB+A map from color and opacity arrays (whose
  // sizes are allowed to differ..)
  auto *colorData = m_colorData->beginAs<math::float3>();
  auto *opacityData = m_opacityData->beginAs<float>();

  size_t tfSize = std::max(m_colorData->size(), m_opacityData->size());

  m_rgbaMap.resize(tfSize);
  for (size_t i = 0; i < tfSize; ++i) {
    float colorPos = tfSize > 1
        ? (float(i) / (tfSize - 1)) * (m_colorData->size() - 1)
        : 0.f;
    float colorFrac = colorPos - floorf(colorPos);

    math::float3 color0 = colorData[int(floorf(colorPos))];
    math::float3 color1 = colorData[int(ceilf(colorPos))];
    math::float3 color = math::float3(math::lerp(color0.x, color1.x, colorFrac),
        math::lerp(color0.y, color1.y, colorFrac),
        math::lerp(color0.z, color1.z, colorFrac));

    float alphaPos = tfSize > 1
        ? (float(i) / (tfSize - 1)) * (m_opacityData->size() - 1)
        : 0.f;
    float alphaFrac = alphaPos - floorf(alphaPos);

    float alpha0 = opacityData[int(floorf(alphaPos))];
    float alpha1 = opacityData[int(ceilf(alphaPos))];
    float alpha = math::lerp(alpha0, alpha1, alphaFrac);

    m_rgbaMap[i] = math::float4(color.x, color.y, color.z, alpha);
  }

  setBarneyParameters();
}

BNVolume TransferFunction1D::createBarneyVolume(BNModel model, int slot)
{
  return m_field
      ? bnVolumeCreate(model, slot, m_field->getBarneyScalarField(model, slot))
      : BNVolume{};
}

box3 TransferFunction1D::bounds() const
{
  return m_bounds;
}

void TransferFunction1D::setBarneyParameters()
{
  BNModel model = trackedModel();
  if (!isValid() || !model || !m_bnVolume)
    return;
  int slot = trackedSlot();

  BNVolume vol = getBarneyVolume(model, slot);
  bnVolumeSetXF(vol,
      (float2 &)m_valueRange,
      (const float4 *)m_rgbaMap.data(),
      m_rgbaMap.size(),
      m_densityScale);
  bnCommit(vol);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Volume *);
