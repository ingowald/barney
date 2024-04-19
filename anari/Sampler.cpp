// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <cassert>

namespace barney_device {

Sampler::Sampler(BarneyGlobalState *s) : Object(ANARI_SAMPLER, s) {}

Sampler::~Sampler() = default;

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

// Subtypes ///////////////////////////////////////////////////////////////////

// Image1D //

Image1D::Image1D(BarneyGlobalState *s) : Sampler(s) {}

Image1D::~Image1D()
{
  cleanup();
}

void Image1D::commit()
{
  cleanup();

  Sampler::commit();

  m_image = getParamObject<helium::Array1D>("image");
  m_inAttribute = toAttribute(getParamString("inAttribute", "attribute0"));
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode = toBarneyAddressMode(getParamString("wrapMode1", "clampToEdge"));
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

void Image1D::setBarneyParameters(BNModel model, BNMaterial mat, int slot) const
{
  if (!m_image)
    return;

  bnSetString(mat, "sampler.type", "image1D");
  bnSet1i(mat, "sampler.image.inAttribute", m_inAttribute);
  bnSet4x4fv(mat, "sampler.image.inTransform", (const float *)&m_inTransform.x);
  bnSet4f(mat,
      "sampler.image.inOffset",
      m_inOffset.x,
      m_inOffset.y,
      m_inOffset.z,
      m_inOffset.w);
  bnSet4x4fv(
      mat, "sampler.image.outTransform", (const float *)&m_outTransform.x);
  bnSet4f(mat,
      "sampler.image.outOffset",
      m_outOffset.x,
      m_outOffset.y,
      m_outOffset.z,
      m_outOffset.w);

  if (!m_texture) {
    m_texture = makeBarneyTexture2D(model,
        slot,
        m_image,
        m_image->size(),
        1,
        m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST,
        m_wrapMode);
  }

  bnSetObject(mat, "sampler.image.image", m_texture);
}

void Image1D::cleanup()
{
  if (m_texture)
    bnRelease(m_texture);
  m_texture = nullptr;
}

// Image2D //

Image2D::Image2D(BarneyGlobalState *s) : Sampler(s) {}

Image2D::~Image2D()
{
  cleanup();
}

void Image2D::commit()
{
  cleanup();

  Sampler::commit();

  m_image = getParamObject<helium::Array2D>("image");
  m_inAttribute = toAttribute(getParamString("inAttribute", "attribute0"));
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

void Image2D::setBarneyParameters(BNModel model, BNMaterial mat, int slot) const
{
  if (!m_image)
    return;

  bnSetString(mat, "sampler.type", "image2D");
  bnSet1i(mat, "sampler.image.inAttribute", m_inAttribute);
  bnSet4x4fv(mat, "sampler.image.inTransform", (const float *)&m_inTransform.x);
  bnSet4f(mat,
      "sampler.image.inOffset",
      m_inOffset.x,
      m_inOffset.y,
      m_inOffset.z,
      m_inOffset.w);
  bnSet4x4fv(
      mat, "sampler.image.outTransform", (const float *)&m_outTransform.x);
  bnSet4f(mat,
      "sampler.image.outOffset",
      m_outOffset.x,
      m_outOffset.y,
      m_outOffset.z,
      m_outOffset.w);

  if (!m_texture) {
    assert(m_wrapMode1 == m_wrapMode2);
    m_texture = makeBarneyTexture2D(model,
        slot,
        m_image,
        m_image->size().x,
        m_image->size().y,
        m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST,
        m_wrapMode1);
  }

  bnSetObject(mat, "sampler.image.image", m_texture);
}

void Image2D::cleanup()
{
  if (m_texture)
    bnRelease(m_texture);
  m_texture = nullptr;
}

// TransformSampler //

TransformSampler::TransformSampler(BarneyGlobalState *s) : Sampler(s) {}

void TransformSampler::commit()
{
  Sampler::commit();

  m_inAttribute = toAttribute(getParamString("inAttribute", "attribute0"));
  m_outTransform = math::identity;
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  getParam("transform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
}

void TransformSampler::setBarneyParameters(
    BNModel model, BNMaterial mat, int slot) const
{
  bnSetString(mat, "sampler.type", "transform");
  bnSet1i(mat, "sampler.transform.inAttribute", m_inAttribute);
  bnSet4x4fv(
      mat, "sampler.transform.outTransform", (const float *)&m_outTransform.x);
  bnSet4f(mat,
      "sampler.transform.outOffset",
      m_outOffset.x,
      m_outOffset.y,
      m_outOffset.z,
      m_outOffset.w);
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
