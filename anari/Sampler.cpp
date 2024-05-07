// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <cassert>
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

  Sampler::Sampler(BarneyGlobalState *s) : Object(ANARI_SAMPLER, s) {}

  Sampler::~Sampler()
  {
    cleanup();
  }

  Sampler *Sampler::createInstance(std::string_view subtype, BarneyGlobalState *s)
  {
    if (subtype == "image1D")
      return new Image1D(s);
    else if (subtype == "image2D")
      return new Image2D(s);
    else if (subtype == "transform")
      return new TransformSampler(s);
    else
      return (Sampler *)new UnknownObject(ANARI_SAMPLER, s);
  }

  // void Sampler::setBarneySampler(BNModel model, int slot, const char *subtype)
  // {
  //   if (!isModelTracked(model, slot)) {
  //     cleanup();
  //     trackModel(model, slot);
  //     m_bnSampler = bnSamplerCreate(model, slot, subtype);
  //     setBarneyParameters();
  //   }
  // }

  void Sampler::cleanup()
  {
    if (m_bnSampler) {
      bnRelease(m_bnSampler);
      m_bnSampler = nullptr;
      bnRelease(m_bnTextureData);
      m_bnTextureData = nullptr;
    }
  }

  // Subtypes ///////////////////////////////////////////////////////////////////

  // Image1D //

  Image1D::Image1D(BarneyGlobalState *s) : Sampler(s) {}

  Image1D::~Image1D() = default;

  void Image1D::commit()
  {
    cleanup();

    Sampler::commit();

    // m_image = getParamObject<helium::Array1D>("image");
    // m_inAttribute = getParamString("inAttribute", "attribute0");
    // m_linearFilter = getParamString("filter", "linear") != "nearest";
    // m_wrapMode = toBarneyAddressMode(getParamString("wrapMode1", "clampToEdge"));
    // m_inTransform = math::identity;
    // getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    // m_inOffset =
    //     getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    // m_outTransform = math::identity;
    // getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    // m_outOffset =
    //     getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    // setBarneyParameters();
  }

  bool Image1D::isValid() const
  {
    return m_image;
  }

  void Image1D::createBarneySampler(BNModel model, int slot)
  {
    // if (!isValid())
    //   return {};
    // setBarneySampler(model, slot, "image1D");
    // return m_bnSampler;
  }

  //   void Image1D::setBarneyParameters()
  // {
  //   if (!m_bnSampler)
  //     return;

  //   // TODO: set and commit parameters on barney sampler
  // }

  // Image2D //

  Image2D::Image2D(BarneyGlobalState *s) : Sampler(s) {}

  Image2D::~Image2D()
  {
    cleanup();
  }

  void Image2D::commit()
  {
    Sampler::commit();

    m_image = getParamObject<helium::Array2D>("image");
    m_inAttribute = getParamString("inAttribute", "attribute0");
    m_linearFilter = getParamString("filter", "linear") != "nearest";
    m_wrapMode1 = toBarneyAddressMode(getParamString("wrapMode1", "clampToEdge"));
    m_wrapMode2 = toBarneyAddressMode(getParamString("wrapMode2", "clampToEdge"));
    m_inTransform = math::identity;
    getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    m_inOffset =
      getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    m_outTransform = math::identity;
    getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    
    //    setBarneyParameters();
    cleanup();
  }

  bool Image2D::isValid() const
  {
    return m_image;
  }

  BNSampler Sampler::getBarneySampler(BNModel model, int slot)
  {
    if (!isValid())
      return {};
    if (!isModelTracked(model, slot)) {
      cleanup();
      trackModel(model, slot);
    }
    if (!m_bnSampler) 
      createBarneySampler(model,slot);
    return m_bnSampler;
  }

  void Image2D::createBarneySampler(BNModel model, int slot) 
  {
    // ------------------------------------------------------------------
    // first, create 2D cuda array of texels. these barney objects
    // SHOULD actually live with their respective image array...
    // ------------------------------------------------------------------
    int width  = m_image->size().x;
    int height = m_image->size().y;
    std::vector<uint32_t> texels;
    if (!convert_to_rgba8(m_image, texels)) {
      std::stringstream ss;
      ss << "unsupported texel type: "
         << anari::toString(m_image->elementType());
      std::string str = ss.str();
      fprintf(stderr, "%s\n", str.c_str());
      texels.resize(width*height);
    }

    if (m_bnTextureData)
      bnRelease(m_bnTextureData);
    m_bnTextureData
      = bnTextureData2DCreate(model,slot,BN_TEXEL_FORMAT_RGBA8,
                              width,height,texels.data());
  
    // ------------------------------------------------------------------
    // now, create sampler over those texels
    // ------------------------------------------------------------------
  
    m_bnSampler = bnSamplerCreate(model,slot,"texture2D");
    bnSetObject(m_bnSampler,"textureData",m_bnTextureData);
  
    BNTextureFilterMode filterMode
      = m_linearFilter
      ? BN_TEXTURE_LINEAR
      : BN_TEXTURE_NEAREST;
  
    bnSet1i(m_bnSampler,"filterMode", (int)filterMode);
    bnSet1i(m_bnSampler,"wrapMode0", (int)m_wrapMode1);
    bnSet1i(m_bnSampler,"wrapMode1", (int)m_wrapMode2);
    bnSet4x4fv(m_bnSampler,"inTransform", (const float *)&m_inTransform);
    bnSet4x4fv(m_bnSampler,"outTransform",(const float *)&m_outTransform);
    bnSet4f(m_bnSampler,"inOffset", m_inOffset.x,m_inOffset.y,m_inOffset.z,m_inOffset.w);
    bnSet4f(m_bnSampler,"outOffset",m_outOffset.x,m_outOffset.y,m_outOffset.z,m_outOffset.w);
    bnSetString(m_bnSampler,"inAttribute",m_inAttribute.c_str());
    bnCommit(m_bnSampler);
  }

  // TransformSampler //

  TransformSampler::TransformSampler(BarneyGlobalState *s) : Sampler(s) {}

  void TransformSampler::commit()
  {
    Sampler::commit();

    m_inAttribute = getParamString("inAttribute", "attribute0");
    m_outTransform = math::identity;
    getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    getParam("transform", ANARI_FLOAT32_MAT4, &m_outTransform);
    m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  }

  void TransformSampler::createBarneySampler(BNModel model, int slot)
  {
    // setBarneySampler(model, slot, "transform");
    // return m_bnSampler;
  }

  // void TransformSampler::setBarneyParameters()
  // {
  //   if (!m_bnSampler)
  //     return;

  //   // TODO: set and commit parameters on barney sampler
  // }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
