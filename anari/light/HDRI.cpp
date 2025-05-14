// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "light/HDRI.h"

namespace barney_device {
  
  HDRILight::HDRILight(BarneyGlobalState *s)
    : Light(s)
  {}

  void HDRILight::commitParameters()
  {
    Light::commitParameters();
    m_scale     = getParam<float>("scale", 1.f);
    m_up        = getParam<math::float3>("up", math::float3(0.f, 0.f, 1.f));
    m_direction = getParam<math::float3>("direction", math::float3(1.f, 0.f, 0.f));
    m_radiance  = getParamObject<helium::Array2D>("radiance");
  }

  void HDRILight::finalize()
  {
    if (!m_radiance) {
      throw std::runtime_error
        ("banari - created hdri light without any radiance values!?");
      return;
    }
    Light::finalize();
  }

  const char *HDRILight::bnSubtype() const
  {
    return "envmap";
  }

  void HDRILight::setBarneyParameters()
  {
    if (!m_bnLight)
      return;

    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;
    
    bnSet3fc(m_bnLight, "direction", m_direction);
    bnSet3fc(m_bnLight, "up",        m_up);
    bnSet1f (m_bnLight, "scale",     m_scale);

    assert(m_radiance);
    int width  = m_radiance->size().x;
    int height = m_radiance->size().y;
    const math::float3 *radianceValues
      = m_radiance->dataAs<math::float3>();
    // cuda textures have to be float4, not float3, so barney only
    // supports float3, too
    std::vector<math::float4> asFloat4(width * height);
    for (int i = 0; i < width * height; i++) {
      (math::float3 &)asFloat4[i] = radianceValues[i];
    }

    BNTexture texture = bnTexture2DCreate(context,slot,
                                          BN_FLOAT4,
                                          width,
                                          height,
                                          asFloat4.data(),
                                          BN_TEXTURE_LINEAR,
                                          BN_TEXTURE_WRAP,
                                          BN_TEXTURE_CLAMP,
                                          BN_COLOR_SPACE_LINEAR);

    bnSetObject(m_bnLight, "texture", texture);
    bnRelease(texture);

    bnCommit(m_bnLight);
  }

} // ::barney_device
