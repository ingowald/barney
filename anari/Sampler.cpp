// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <cassert>
#include <iostream>

namespace tally_device {

  Sampler::Sampler(TallyGlobalState *s) : Object(ANARI_SAMPLER, s) {}

  Sampler::~Sampler()
  {
    cleanup();
  }

  Sampler *Sampler::createInstance(std::string_view subtype,
                                   TallyGlobalState *s)
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

  void Sampler::cleanup()
  {
    if (m_bnSampler) {
      // bnRelease(m_bnSampler);
      m_bnSampler = nullptr;
    }
    if (m_bnTextureData) {
      // bnRelease(m_bnTextureData);
      m_bnTextureData = nullptr;
    }
  }

  // Subtypes ///////////////////////////////////////////////////////////////////

  // Image1D //

  Image1D::Image1D(TallyGlobalState *s) : Sampler(s) {}

  Image1D::~Image1D() = default;

  void Image1D::commit()
  {
    cleanup();

    Sampler::commit();

    // m_image = getParamObject<helium::Array1D>("image");
    // m_inAttribute = getParamString("inAttribute", "attribute0");
    // m_linearFilter = getParamString("filter", "linear") != "nearest";
    // m_wrapMode = toTallyAddressMode(getParamString("wrapMode1", "clampToEdge"));
    // m_inTransform = math::identity;
    // getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    // m_inOffset =
    //     getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    // m_outTransform = math::identity;
    // getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    // m_outOffset =
    //     getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    // setTallyParameters();
  }

  bool Image1D::isValid() const
  {
    return m_image;
  }

  void Image1D::createTallySampler(TallyModel::SP model, int slot)
  {
  }

  // Image2D //

  Image2D::Image2D(TallyGlobalState *s) : Sampler(s) {}

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
#if TALLY
    m_wrapMode1 = toTallyAddressMode(getParamString("wrapMode1", "clampToEdge"));
    m_wrapMode2 = toTallyAddressMode(getParamString("wrapMode2", "clampToEdge"));
#endif
    m_inTransform = math::identity;
    getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    m_inOffset =
      getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    m_outTransform = math::identity;
    getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  }

  bool Image2D::isValid() const
  {
    return m_image;
  }

  TallySampler::SP Sampler::getTallySampler(TallyModel::SP model, int slot)
  {
    if (!isValid())
      return {};
    if (!isModelTracked(model, slot)) {
      cleanup();
      trackModel(model, slot);
    }
    if (!m_bnSampler) 
      createTallySampler(model,slot);
    return m_bnSampler;
  }

  void Image2D::createTallySampler(TallyModel::SP model, int slot) 
  {
    // ------------------------------------------------------------------
    // first, create 2D cuda array of texels. these tally objects
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

#if TALLY
    if (m_bnTextureData)
      bnRelease(m_bnTextureData);
    m_bnTextureData
      = bnTextureData2DCreate(model,slot,BN_TEXEL_FORMAT_RGBA8,
                              width,height,texels.data());
#endif
    // ------------------------------------------------------------------
    // now, create sampler over those texels
    // ------------------------------------------------------------------
  
    m_bnSampler = TallySampler::create("texture2D");//bnSamplerCreate(model,slot,"texture2D");
#if TALLY
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
#endif
  }

  // TransformSampler //

  TransformSampler::TransformSampler(TallyGlobalState *s) : Sampler(s) {}

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

  void TransformSampler::createTallySampler(TallyModel::SP model, int slot)
  {
    // setTallySampler(model, slot, "transform");
    // return m_bnSampler;
  }

  // void TransformSampler::setTallyParameters()
  // {
  //   if (!m_bnSampler)
  //     return;

  //   // TODO: set and commit parameters on tally sampler
  // }

} // namespace tally_device

TALLY_ANARI_TYPEFOR_DEFINITION(tally_device::Sampler *);
