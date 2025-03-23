// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "light/Point.h"

namespace barney_device {
  
  PointLight::PointLight(BarneyGlobalState *s) : Light(s) {}

  void PointLight::commitParameters()
  {
    Light::commitParameters();
    m_power = getParam<float>("power", 1.f);
    m_intensity = getParam<float>("intensity", NAN);
  }

  const char *PointLight::bnSubtype() const
  {
    return "point";
  }

  void PointLight::setBarneyParameters()
  {
    if (!m_bnLight)
      return;
    bnSet3fc(m_bnLight, "direction", m_position);
    bnSet3fc(m_bnLight, "color", m_color);
    bnSet1f(m_bnLight, "intensity", m_intensity);
    bnSet1f(m_bnLight, "power", m_power);
    bnCommit(m_bnLight);
  }

} // ::barney_device
