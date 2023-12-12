// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Volume.h"
// std
#include <numeric>

namespace barney_device {

Volume::Volume(BarneyGlobalState *s) : Object(ANARI_VOLUME, s)
{
  s->objectCounts.volumes++;
}

Volume::~Volume()
{
  deviceState()->objectCounts.volumes--;
}

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

// Subtypes ///////////////////////////////////////////////////////////////////

TransferFunction1D::TransferFunction1D(BarneyGlobalState *s) : Volume(s) {}

void TransferFunction1D::commit()
{
  Volume::commit();

  cleanup();

  m_field = getParamObject<SpatialField>("field");
  if (!m_field) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "no spatial field provided to transferFunction1D volume");
    return;
  }

  m_bounds = m_field->bounds();

  m_valueRange = getParam<anari::box1>("valueRange", anari::box1{0.f, 1.f});

  m_colorData = getParamObject<helium::Array1D>("color");
  m_opacityData = getParamObject<helium::Array1D>("opacity");
  m_densityScale = getParam<float>("densityScale", 1.f);

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
  auto *colorData = m_colorData->beginAs<float3>();
  auto *opacityData = m_opacityData->beginAs<float>();

  size_t tfSize = std::max(m_colorData->size(), m_opacityData->size());

  m_rgbaMap.resize(tfSize);
  for (size_t i = 0; i < tfSize; ++i) {
    float colorPos = tfSize > 1
        ? (float(i) / (tfSize - 1)) * (m_colorData->size() - 1)
        : 0.f;
    float colorFrac = colorPos - floorf(colorPos);

    float3 color0 = colorData[int(floorf(colorPos))];
    float3 color1 = colorData[int(ceilf(colorPos))];
    float3 color = make_float3(linalg::lerp(color0.x, color1.x, colorFrac),
        linalg::lerp(color0.y, color1.y, colorFrac),
        linalg::lerp(color0.z, color1.z, colorFrac));

    float alphaPos = tfSize > 1
        ? (float(i) / (tfSize - 1)) * (m_opacityData->size() - 1)
        : 0.f;
    float alphaFrac = alphaPos - floorf(alphaPos);

    float alpha0 = opacityData[int(floorf(alphaPos))];
    float alpha1 = opacityData[int(ceilf(alphaPos))];
    float alpha = linalg::lerp(alpha0, alpha1, alphaFrac);

    m_rgbaMap[i] = make_float4(color.x, color.y, color.z, alpha);
  }
}

BNVolume TransferFunction1D::makeBarneyVolume(BNDataGroup dg) const
{
  auto ctx = deviceState()->context;
  static BNVolume bnVol{nullptr}; // TODO: really find out if volume has changed!
  if (!bnVol)
    bnVol = bnVolumeCreate(dg, m_field->makeBarneyScalarField(dg));
  bnVolumeSetXF(bnVol,
      (float2 &)m_valueRange,
      m_rgbaMap.data(),
      m_rgbaMap.size(),
      m_densityScale);
  return bnVol;
}

anari::box3 TransferFunction1D::bounds() const
{
  return m_bounds;
}

void TransferFunction1D::cleanup() {}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Volume *);
