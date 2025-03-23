// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "light/Light.h"

namespace barney_device {

  struct Directional : public Light
  {
    Directional(BarneyGlobalState *s);

    void commitParameters() override;

  private:
    const char *bnSubtype() const override;
    void setBarneyParameters() override;

    /*! SPEC: main emission direction of the directional light */
    math::float3 m_direction{0.f, 0.f, -1.f};

    /*! SPEC: the amount of light arriving at a surface point,
      assuming the light is oriented towards to the surface, in
      W/m2 */
    float m_irradiance = NAN;
    /*! the amount of light emitted in a direction, in W/sr/m2;
      irradiance takes precedence if also specified */
    float m_radiance = 1.f;
  };

} // ::barney_device
