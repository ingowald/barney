// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "Array.h"

namespace barney_device {

  struct Renderer : public Object
  {
    Renderer(BarneyGlobalState *s);
    ~Renderer() override;

    void commit() override;

    // int pixelSamples() const;
    // float radiance() const;
    bool crosshairs() const;
    
    bool isValid() const override;

    BNRenderer barneyRenderer = 0;
  private:
    int    m_pixelSamples{8*16};
    float  m_ambientRadiance{0.8f};
    bool   m_crosshairs{false};
    anari::math::float4 m_background{0.f,0.f,0.f,1.f};
    helium::ChangeObserverPtr<Array2D> m_backgroundImage;
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Renderer *, ANARI_RENDERER);
