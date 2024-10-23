// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Renderer.h"

namespace tally_device {

Renderer::Renderer(TallyGlobalState *s) : Object(ANARI_RENDERER, s) {}

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

} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::Renderer *);
