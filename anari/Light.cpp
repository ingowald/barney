// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
#include <iostream>

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
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
    PING; PRINT(type);
    if (type == "directional")
      return new Directional(s);
    else if (type == "hdri")
      return new HDRILight(s);
    else
      return (Light *)new UnknownObject(ANARI_LIGHT, s);
  }

  void Light::markCommitted()
  {
    // NOTE: shouldn't need to override this to cause a BNModel rebuild...
    deviceState()->markSceneChanged();
    Object::markCommitted();
  }

  void Light::commit()
  {
    m_radiance = getParam<math::float3>("color", math::float3(1.f, 1.f, 1.f));
  }

  BNLight Light::getBarneyLight(BNModel model, int slot)
  {
    if (!isModelTracked(model, slot)) {
      cleanup();
      trackModel(model, slot);
      m_bnLight = bnLightCreate(model, slot, bnSubtype());
      setBarneyParameters();
    }

    return m_bnLight;
  }

  void Light::cleanup()
  {
    if (m_bnLight)
      bnRelease(m_bnLight);
    m_bnLight = nullptr;
  }

  // // Subtypes ///////////////////////////////////////////////////////////////////

  // Directional::Directional(BarneyGlobalState *s) : Light(s) {}

  // void Directional::commit()
  // {
  //   Light::commit();
  //   m_radiance *= getParam<float>("irradiance", 1.f);
  //   m_dir = getParam<math::float3>("direction", math::float3(0.f, 0.f, -1.f));
  //   setBarneyParameters();
  // }

  // const char *Directional::bnSubtype() const
  // {
  //   return "directional";
  // }

  // void Light::cleanup()
  // {
  //   if (m_bnLight)
  //     bnRelease(m_bnLight);
  //   m_bnLight = nullptr;
  // }

  // Subtypes ///////////////////////////////////////////////////////////////////

  Directional::Directional(BarneyGlobalState *s) : Light(s) {}

  void Directional::commit()
  {
    Light::commit();
    m_radiance *= getParam<float>("irradiance", 1.f);
    m_dir = getParam<math::float3>("direction", math::float3(0.f, 0.f, -1.f));
    setBarneyParameters();
  }

  const char *Directional::bnSubtype() const
  {
    return "directional";
  }

  void Directional::setBarneyParameters() 
  {
    if (!m_bnLight)
      return;
    bnSet3fc(m_bnLight, "direction", (const float3 &)m_dir);
    bnSet3fc(m_bnLight, "radiance", (const float3 &)m_radiance);
    bnCommit(m_bnLight);
  }


  // Subtypes ///////////////////////////////////////////////////////////////////

  HDRILight::HDRILight(BarneyGlobalState *s) : Light(s) {}
  
  void HDRILight::commit()
  {
    std::cout << "#banari: creating hdri light " << std::endl;
    Light::commit();
    m_up = getParam<math::float3>("up", math::float3(0.f,0.f,1.f));
    m_direction = getParam<math::float3>("direction", math::float3(1.f,0.f,0.f));
    m_radiance = getParamObject<helium::Array2D>("radiance");

    if (!m_radiance)
      throw std::runtime_error("banari - created hdri light without any radiance values!?");
  // int numVertices = m_vertexPosition->totalSize();
  // int numIndices = m_index ? m_index->size() : (m_generatedIndices.size() / 3);
    
    setBarneyParameters();
  }

  const char *HDRILight::bnSubtype() const
  {
    return "directional";
  }

  void HDRILight::setBarneyParameters() 
  {
    if (!m_bnLight)
      return;
    BNModel model = trackedModel();
    int slot = trackedSlot();

    bnSet3fc(m_bnLight, "direction", (const float3 &)m_direction);
    bnSet3fc(m_bnLight, "up", (const float3 &)m_up);
    const math::float3 *radianceValues = (const math::float3 *)m_radiance->data();
    int width  = m_radiance->size().x;
    int height = m_radiance->size().y;
    BNData radianceData
      = bnDataCreate(model,slot,BN_FLOAT3,
                     width*height,radianceData);
    bnSetData(m_bnLight, "texture.values", radianceData);
    bnSet2i(m_bnLight,"texture.dims",width,height);
    bnCommit(m_bnLight);
    bnRelease(radianceData);
  }
  
} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Light *);
