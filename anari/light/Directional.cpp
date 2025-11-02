// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "light/Directional.h"

namespace barney_device {
  
  Directional::Directional(BarneyGlobalState *s) : Light(s) {}

  void Directional::commitParameters()
  {
    Light::commitParameters();
    m_irradiance = getParam<float>("irradiance", NAN);
    m_radiance = getParam<float>("radiance", 1.f);
    m_direction =
      getParam<math::float3>("direction", math::float3(0.f, 0.f, -1.f));
  }

  const char *Directional::bnSubtype() const
  {
    return "directional";
  }

  void Directional::setBarneyParameters()
  {
    if (!m_bnLight)
      return;
    bnSetVec(m_bnLight, "direction", m_direction);
    bnSetVec(m_bnLight, "color", m_color);
    bnSet1f (m_bnLight, "radiance", m_radiance);
    bnSet1f (m_bnLight, "irradiance", m_irradiance);
    bnCommit(m_bnLight);
  }

} // ::barney_device
