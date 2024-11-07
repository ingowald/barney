// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Renderer.h"

namespace barney_device {

Renderer::Renderer(BarneyGlobalState *s)
    : Object(ANARI_RENDERER, s), m_backgroundImage(this)
{
  barneyRenderer = bnRendererCreate(deviceState()->context, "default");
}

Renderer::~Renderer()
{
  bnRelease(barneyRenderer);
}

void Renderer::commit()
{
  m_pixelSamples = getParam<int>("pixelSamples", 16);
  m_ambientRadiance = getParam<float>("ambientRadiance", .8f);
  m_crosshairs = getParam<bool>("crosshairs", false);
  m_background = getParam<math::float4>("background", math::float4(0, 0, 0, 1));
  m_backgroundImage = getParamObject<Array2D>("background");

  bnSet4fc(barneyRenderer, "bgColor", (const float4 &)m_background);
  bnSet1i(barneyRenderer, "crosshairs", (int)m_crosshairs);
  bnSet1i(barneyRenderer, "pathsPerPixel", (int)m_pixelSamples);
  bnSet1f(barneyRenderer, "ambientRadiance", m_ambientRadiance);
}

bool Renderer::crosshairs() const
{
  return m_crosshairs;
}

bool Renderer::isValid() const
{
  return barneyRenderer != 0;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Renderer *);
