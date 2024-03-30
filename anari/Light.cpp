// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"

namespace barney_device {

Light::Light(BarneyGlobalState *s) : Object(ANARI_CAMERA, s) {}

Light::~Light() = default;

Light *Light::createInstance(std::string_view type, BarneyGlobalState *s)
{
  if (type == "directional")
    return new Directional(s);
  else
    return (Light *)new UnknownObject(ANARI_CAMERA, s);
}

void Light::commit()
{
  m_radiance = getParam<float>("radiance", 1.f)
      * getParam<math::float3>("color", math::float3(1.f, 1.f, 1.f));
  markUpdated();
}

// Subtypes ///////////////////////////////////////////////////////////////////

Directional::Directional(BarneyGlobalState *s) : Light(s) {}

void Directional::commit()
{
  Light::commit();
  m_dir = getParam<math::float3>("direction", math::float3(0.f, 0.f, -1.f));
}

void Directional::setBarneyParameters(BNLight light) const
{
  bnSet3fc(light, "direction", (const float3 &)m_dir);
  bnSet3fc(light, "radiance", (const float3 &)m_radiance);
  bnCommit(light);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Light *);
