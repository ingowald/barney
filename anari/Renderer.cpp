// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Renderer.h"

namespace barney_device {

Renderer::Renderer(BarneyGlobalState *s) : Object(ANARI_RENDERER, s) {}

Renderer::~Renderer() = default;

void Renderer::commit()
{
  m_pixelSamples = getParam<int>("pixelSamples", 16);
  m_radiance = getParam<float>("ambientRadiance", .8f);
  m_crosshairs = getParam<bool>("crosshairs", false);
}

int Renderer::pixelSamples() const
{
  return m_pixelSamples;
}

float Renderer::radiance() const
{
  return m_radiance;
}

bool Renderer::crosshairs() const
{
  return m_crosshairs;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Renderer *);
