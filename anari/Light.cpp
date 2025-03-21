// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
// std
#include <iostream>

#ifndef PRINT
#define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
#define PING                                                                   \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__            \
            << std::endl;
#else
#define PING                                                                   \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__     \
            << std::endl;
#endif
#endif

namespace barney_device {

Light::Light(BarneyGlobalState *s) : Object(ANARI_LIGHT, s) {}

Light::~Light()
{
  cleanup();
}

Light *Light::createInstance(std::string_view type, BarneyGlobalState *s)
{
  if (type == "directional")
    return new Directional(s);
  else if (type == "hdri")
    return new HDRILight(s);
  else
    return (Light *)new UnknownObject(ANARI_LIGHT, s);
}

void Light::markFinalized()
{
  // NOTE: shouldn't need to override this to cause a BNContext rebuild...
  deviceState()->markSceneChanged();
  Object::markFinalized();
}

void Light::commitParameters()
{
  m_color = getParam<math::float3>("color", math::float3(1.f, 1.f, 1.f));
}

void Light::finalize()
{
  setBarneyParameters();
}

BNLight Light::getBarneyLight(BNContext context)
{
  m_bnLight = bnLightCreate(getContext(), 0, bnSubtype());
  setBarneyParameters();
  return m_bnLight;
}

void Light::cleanup()
{
  if (m_bnLight)
    bnRelease(m_bnLight);
  m_bnLight = nullptr;
}

// Subtypes ///////////////////////////////////////////////////////////////////

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

// Subtypes ///////////////////////////////////////////////////////////////////

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
  bnSet3fc(m_bnLight, "direction", m_direction);
  bnSet3fc(m_bnLight, "color", m_color);
  bnSet1f(m_bnLight, "radiance", m_radiance);
  bnSet1f(m_bnLight, "irradiance", m_irradiance);
  bnCommit(m_bnLight);
}

// Subtypes ///////////////////////////////////////////////////////////////////

HDRILight::HDRILight(BarneyGlobalState *s) : Light(s) {}

void HDRILight::commitParameters()
{
  Light::commitParameters();
  m_up = getParam<math::float3>("up", math::float3(0.f, 0.f, 1.f));
  m_direction =
      getParam<math::float3>("direction", math::float3(1.f, 0.f, 0.f));
  m_radiance = getParamObject<helium::Array2D>("radiance");
}

void HDRILight::finalize()
{
  if (!m_radiance) {
    throw std::runtime_error(
        "banari - created hdri light without any radiance values!?");
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

  bnSet3fc(m_bnLight, "direction", m_direction);
  bnSet3fc(m_bnLight, "up", m_up);

  assert(m_radiance);
  int width = m_radiance->size().x;
  int height = m_radiance->size().y;
  const math::float3 *radianceValues
      // = (const math::float3 *)m_radiance->data();
      = m_radiance->dataAs<math::float3>();
  // cuda textures have to be float4, not float3, so barney only
  // supports float3, too
  std::vector<math::float4> asFloat4(width * height);
  for (int i = 0; i < width * height; i++) {
    (math::float3 &)asFloat4[i] = radianceValues[i];
  }

  BNTexture texture = bnTexture2DCreate(getContext(),
      0, // model,slot,
      BN_FLOAT4_RGBA,
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

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Light *);
