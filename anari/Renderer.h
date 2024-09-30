// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace barney_device {

struct Renderer : public Object
{
  Renderer(BarneyGlobalState *s);
  ~Renderer() override;

  void commit() override;

  int pixelSamples() const;
  float radiance() const;
  bool crosshairs() const;

 private:
  int m_pixelSamples{8*16};
  float m_radiance{0.8f};
  bool m_crosshairs{false};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Renderer *, ANARI_RENDERER);
