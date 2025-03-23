// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "light/Light.h"

namespace barney_device {

  struct PointLight : public Light
  {
    PointLight(BarneyGlobalState *s);

    void commitParameters() override;

  private:
    const char *bnSubtype() const override;
    void setBarneyParameters() override;

    math::float3 m_position{0.f, 0.f, 0.f};

    /*! SPEC: the overall amount of light emitted by the light in a
      direction, in W/sr */
    float m_intensity = NAN;

    /*! SPEC: the overall amount of light energy emitted, in W;
      intensity takes precedence if also specified */
    float m_power = 1.f;
  };
  
} // ::barney_device
