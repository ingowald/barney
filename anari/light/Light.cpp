// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "light/Light.h"
#include "light/Directional.h"
#include "light/Point.h"
#include "light/HDRI.h"

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
    // else if (type == "point")
    //   return new PointLight(s);
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

  BNLight Light::getBarneyLight()
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;
    
    m_bnLight = bnLightCreate(context,slot,bnSubtype());
    setBarneyParameters();
    return m_bnLight;
  }

  void Light::cleanup()
  {
    if (m_bnLight)
      bnRelease(m_bnLight);
    m_bnLight = nullptr;
  }

} // ::barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Light *);
