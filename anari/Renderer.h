// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"

namespace barney_device {

struct Renderer : public Object
{
  Renderer(BarneyGlobalState *s);
  ~Renderer() override;

  void commitParameters() override;
  void finalize() override;

  bool crosshairs() const;
  bool isValid() const override;

  BNRenderer barneyRenderer{nullptr};

 private:
  BNTexture2D barneyBackgroundImage{nullptr};

  int m_pixelSamples{8 * 16};
  float m_ambientRadiance{0.8f};
  bool m_crosshairs{false};
  anari::math::float4 m_background{0.f, 0.f, 0.f, 1.f};
  helium::ChangeObserverPtr<Array2D> m_backgroundImage;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Renderer *, ANARI_RENDERER);
