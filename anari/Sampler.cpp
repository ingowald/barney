// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <cassert>

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

void Sampler::setBarneySampler(BNModel model, int slot, const char *subtype)
{
  if (!isModelTracked(model, slot)) {
    cleanup();
    trackModel(model, slot);
#if 0 // bnSamplerCreate() symbol not defined??
    m_bnSampler = bnSamplerCreate(model, slot, subtype);
#endif
    setBarneyParameters();
  }
}

void Sampler::cleanup()
{
  if (m_bnSampler) {
    bnRelease(m_bnSampler);
    m_bnSampler = nullptr;
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
  setBarneyParameters();
}

bool Image1D::isValid() const
{
  return m_image;
}

BNSampler Image1D::getBarneySampler(BNModel model, int slot)
{
  if (!isValid())
    return {};
  setBarneySampler(model, slot, "image1D");
  return m_bnSampler;
}

void Image1D::setBarneyParameters()
{
  if (!m_bnSampler)
    return;

  // TODO: set and commit parameters on barney sampler
}

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
  setBarneyParameters();
}

bool Image2D::isValid() const
{
  return m_image;
}

BNSampler Image2D::getBarneySampler(BNModel model, int slot)
{
  if (!isValid())
    return {};
  setBarneySampler(model, slot, "image2D");
  return m_bnSampler;
}

void Image2D::setBarneyParameters()
{
  if (!m_bnSampler)
    return;

  // TODO: set and commit parameters on barney sampler
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

BNSampler TransformSampler::getBarneySampler(BNModel model, int slot)
{
  setBarneySampler(model, slot, "transform");
  return m_bnSampler;
}

void TransformSampler::setBarneyParameters()
{
  if (!m_bnSampler)
    return;

  // TODO: set and commit parameters on barney sampler
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
