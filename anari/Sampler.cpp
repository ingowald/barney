// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"

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

void Image1D::commit()
{
  Sampler::commit();

  m_image = getParamObject<helium::Array1D>("image");
  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode = toWrapMode(getParamString("wrapMode1", "clampToEdge"));
  getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
  m_inOffset = getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset = getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
}

// Image2D //

Image2D::Image2D(BarneyGlobalState *s) : Sampler(s) {}

void Image2D::commit()
{
  Sampler::commit();

  m_image = getParamObject<helium::Array2D>("image");
  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode1 = toWrapMode(getParamString("wrapMode1", "clampToEdge"));
  m_wrapMode2 = toWrapMode(getParamString("wrapMode2", "clampToEdge"));
  getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
  m_inOffset = getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset = getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
}

// TransformSampler //

TransformSampler::TransformSampler(BarneyGlobalState *s) : Sampler(s) {}

void TransformSampler::commit()
{
  Sampler::commit();

  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  m_outTransform = math::identity;
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  getParam("transform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset = getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
