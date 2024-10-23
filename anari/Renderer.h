// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace tally_device {

struct Renderer : public Object
{
  Renderer(TallyGlobalState *s);
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

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Renderer *, ANARI_RENDERER);
