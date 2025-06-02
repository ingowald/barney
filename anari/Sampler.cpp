// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <cassert>
#include <iostream>

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

void Sampler::cleanup()
{
  if (m_bnSampler) {
    bnRelease(m_bnSampler);
    m_bnSampler = nullptr;
  }
  if (m_bnTextureData) {
    bnRelease(m_bnTextureData);
    m_bnTextureData = nullptr;
  }
}

// Subtypes ///////////////////////////////////////////////////////////////////

// Image1D //

Image1D::Image1D(BarneyGlobalState *s) : Sampler(s) {}

Image1D::~Image1D() = default;

void Image1D::commitParameters()
{
  Sampler::commitParameters();
  m_image = getParamObject<helium::Array1D>("image");
  m_inAttribute = getParamString("inAttribute", "attribute0");
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode = toBarneyAddressMode(getParamString("wrapMode", "clampToEdge"));
  m_inTransform = math::identity;
  getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
  m_inOffset =
      getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  m_outTransform = math::identity;
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
}

bool Image1D::isValid() const
{
  return m_image;
}

void Image1D::createBarneySampler()
{
  int slot = deviceState()->slot;
  auto context = deviceState()->tether->context;
  if (m_bnTextureData)
    bnRelease(m_bnTextureData);
  // ------------------------------------------------------------------
  // first, create 2D cuda array of texels. these barney objects
  // SHOULD actually live with their respective image array...
  // ------------------------------------------------------------------
  int width = (int)m_image->size();
  int height = 1;
  std::vector<uint32_t> texels;
  if (!convert_to_rgba8(m_image, texels)) {
    std::stringstream ss;
    ss << "unsupported texel type: " << anari::toString(m_image->elementType());
    std::string str = ss.str();
    fprintf(stderr, "%s\n", str.c_str());
    texels.resize(width * height);
  }
  if (m_image->elementType() == ANARI_FLOAT32_VEC4) {
    const anari::math::float4 *colors =
        (const anari::math::float4 *)m_image->data();
    m_bnTextureData =
        bnTextureData2DCreate(context, slot, BN_FLOAT4, width, height, colors);
  } else {
    m_bnTextureData = bnTextureData2DCreate(
        context, slot, BN_UFIXED8_RGBA, width, height, texels.data());
  }

  m_bnSampler = bnSamplerCreate(context, slot, "texture2D");
  bnSetObject(m_bnSampler, "textureData", m_bnTextureData);

  // ------------------------------------------------------------------
  // now, create sampler over those texels
  // ------------------------------------------------------------------

  BNTextureFilterMode filterMode =
      m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST;

  bnSet1i(m_bnSampler, "filterMode", (int)filterMode);
  bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode);
  bnSet1i(m_bnSampler, "wrapMode1", (int)BN_TEXTURE_CLAMP);
  bnSet4x4fv(m_bnSampler, "inTransform", (const bn_float4 *)&m_inTransform);
  bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
  bnSet4f(m_bnSampler,
      "inOffset",
      m_inOffset.x,
      m_inOffset.y,
      m_inOffset.z,
      m_inOffset.w);
  bnSet4f(m_bnSampler,
      "outOffset",
      m_outOffset.x,
      m_outOffset.y,
      m_outOffset.z,
      m_outOffset.w);
  bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
  bnCommit(m_bnSampler);
}

// Image2D //

Image2D::Image2D(BarneyGlobalState *s) : Sampler(s) {}

Image2D::~Image2D() = default;

void Image2D::commitParameters()
{
  Sampler::commitParameters();
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
}

bool Image2D::isValid() const
{
  return m_image;
}

BNSampler Sampler::getBarneySampler()
{
  if (!isValid())
    return {};
  if (!m_bnSampler)
    createBarneySampler();
  return m_bnSampler;
}

void Image2D::createBarneySampler()
{
  int slot = deviceState()->slot;
  auto context = deviceState()->tether->context;

  // ------------------------------------------------------------------
  // first, create 2D cuda array of texels. these barney objects
  // SHOULD actually live with their respective image array...
  // ------------------------------------------------------------------
  int width = m_image->size().x;
  int height = m_image->size().y;
  if (m_bnTextureData)
    bnRelease(m_bnTextureData);

  if (m_image->elementType() == ANARI_FLOAT32_VEC4) {
    const anari::math::float4 *colors =
        (const anari::math::float4 *)m_image->data();
    m_bnTextureData =
        bnTextureData2DCreate(context, slot, BN_FLOAT4, width, height, colors);
  } else {
    std::vector<uint32_t> texels;
    if (!convert_to_rgba8(m_image, texels)) {
      std::stringstream ss;
      ss << "unsupported texel type: "
         << anari::toString(m_image->elementType());
      std::string str = ss.str();
      fprintf(stderr, "%s\n", str.c_str());
      texels.resize(width * height);
    }
    m_bnTextureData = bnTextureData2DCreate(
        context, slot, BN_UFIXED8_RGBA, width, height, texels.data());
  }

  m_bnSampler = bnSamplerCreate(context, slot, "texture2D");
  BNTextureFilterMode filterMode =
      m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST;
  bnSet1i(m_bnSampler, "filterMode", (int)filterMode);
  bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode1);
  bnSet1i(m_bnSampler, "wrapMode1", (int)m_wrapMode2);
  bnSet4x4fv(m_bnSampler, "inTransform", (const bn_float4 *)&m_inTransform);
  bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
  bnSet4f(m_bnSampler,
      "inOffset",
      m_inOffset.x,
      m_inOffset.y,
      m_inOffset.z,
      m_inOffset.w);
  bnSet4f(m_bnSampler,
      "outOffset",
      m_outOffset.x,
      m_outOffset.y,
      m_outOffset.z,
      m_outOffset.w);
  bnSetObject(m_bnSampler, "textureData", m_bnTextureData);
  bnCommit(m_bnSampler);
}

/// Transform ///
TransformSampler::TransformSampler(BarneyGlobalState *s) : Sampler(s) {}

TransformSampler::~TransformSampler() = default;

void TransformSampler::commitParameters()
{
  Sampler::commitParameters();
  m_outTransform = math::identity;
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  m_inAttribute = getParamString("inAttribute", "attribute0");
}

void TransformSampler::createBarneySampler()
{
  int slot = deviceState()->slot;
  auto context = deviceState()->tether->context;
  m_bnSampler = bnSamplerCreate(context, slot, "transform");
  bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
  bnSet4f(m_bnSampler,
      "outOffset",
      m_outOffset.x,
      m_outOffset.y,
      m_outOffset.z,
      m_outOffset.w);
  bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
  bnCommit(m_bnSampler);
  bnCommit(m_bnSampler);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
