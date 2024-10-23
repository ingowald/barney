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

namespace tally_device {

  Light::Light(TallyGlobalState *s) : Object(ANARI_LIGHT, s) {}

  Light::~Light()
  {
    cleanup();
  }

  Light *Light::createInstance(std::string_view type, TallyGlobalState *s)
  {
    if (type == "directional")
      return new Directional(s);
    else if (type == "hdri")
      return new HDRILight(s);
    else
      return (Light *)new UnknownObject(ANARI_LIGHT, s);
  }

  void Light::markCommitted()
  {
    // NOTE: shouldn't need to override this to cause a TallyModel::SP rebuild...
    deviceState()->markSceneChanged();
    Object::markCommitted();
  }

  void Light::commit()
  {
    m_radiance = getParam<math::float3>("color", math::float3(1.f, 1.f, 1.f));
  }

  TallyLight::SP Light::getTallyLight(TallyModel::SP model, int slot)
  {
    if (!isModelTracked(model, slot)) {
      cleanup();
      trackModel(model, slot);
      m_bnLight = TallyLight::create(bnSubtype());// = bnLightCreate(model, slot, bnSubtype());
      setTallyParameters();
    }

    return m_bnLight;
  }

  void Light::cleanup()
  {
    // if (m_bnLight)
    //   bnRelease(m_bnLight);
    m_bnLight = nullptr;
  }

  // Subtypes ///////////////////////////////////////////////////////////////////

  Directional::Directional(TallyGlobalState *s) : Light(s) {}

  void Directional::commit()
  {
    Light::commit();
    m_radiance *= getParam<float>("irradiance", 1.f);
    m_dir = getParam<math::float3>("direction", math::float3(0.f, 0.f, -1.f));
    setTallyParameters();
  }

  const char *Directional::bnSubtype() const
  {
    return "directional";
  }

  void Directional::setTallyParameters() 
  {
    if (!m_bnLight)
      return;
#if TALLY
    bnSet3fc(m_bnLight, "direction", (const float3 &)m_dir);
    bnSet3fc(m_bnLight, "radiance", (const float3 &)m_radiance);
    bnCommit(m_bnLight);
#endif
  }


  // Subtypes ///////////////////////////////////////////////////////////////////

  HDRILight::HDRILight(TallyGlobalState *s) : Light(s) {}
  
  void HDRILight::commit()
  {
    std::cout << "#banari: creating hdri light " << std::endl;
    Light::commit();
    m_up
      = getParam<math::float3>("up", math::float3(0.f,0.f,1.f));
    m_direction
      = getParam<math::float3>("direction", math::float3(1.f,0.f,0.f));
    m_radiance
      = getParamObject<helium::Array2D>("radiance");

    if (!m_radiance)
      throw std::runtime_error("banari - created hdri light without any radiance values!?");
  // int numVertices = m_vertexPosition->totalSize();
  // int numIndices = m_index ? m_index->size() : (m_generatedIndices.size() / 3);
    
    setTallyParameters();
  }

  const char *HDRILight::bnSubtype() const
  {
    return "envmap";
  }

  void HDRILight::setTallyParameters() 
  {
    if (!m_bnLight)
      return;
#if TALLY
    TallyModel::SP model = trackedModel();
    int slot = trackedSlot();

    bnSet3fc(m_bnLight, "direction", (const float3 &)m_direction);
    bnSet3fc(m_bnLight, "up", (const float3 &)m_up);

    assert(m_radiance);
    int width  = m_radiance->size().x;
    int height = m_radiance->size().y;
    const math::float3 *radianceValues
      // = (const math::float3 *)m_radiance->data();
      = m_radiance->dataAs<math::float3>();
    std::vector<math::float4> asFloat4(width*height);
    for (int i=0;i<width*height;i++) 
      (math::float3&)asFloat4[i] = radianceValues[i];

    BNTexture texture
      = bnTexture2DCreate(model,slot,BN_TEXEL_FORMAT_RGBA32F,
                          width,height,asFloat4.data());
                                          
    bnSetObject(m_bnLight, "texture", texture);
    bnRelease(texture);

    bnCommit(m_bnLight);
#endif
  }
  
} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::Light *);
