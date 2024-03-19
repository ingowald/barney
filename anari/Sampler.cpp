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
  else
    return (Sampler *)new UnknownObject(ANARI_SAMPLER, s);
}

// Subtypes ///////////////////////////////////////////////////////////////////

// Image1D //

Image1D::Image1D(BarneyGlobalState *s) : Sampler(s) {}

void Image1D::commit()
{
  m_image = getParamObject<Array1D>("image");
  m_inAttribute = getParamString("inAttribute", "attribute0");
  m_linearFilter = getParamString("filter", "linear") != "nearest";
  m_wrapMode = getParamString("wrapMode1", "clampToEdge");
  m_inTransform =
      getParam<math::mat4>("inTransform", math::mat4(linalg::identity));
  m_inOffset =
      getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  m_outTransform =
      getParam<math::mat4>("outTransform", math::mat4(linalg::identity));
  m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));

  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image1D sampler");
  }
}

// Image2D //

Image2D::Image2D(BarneyGlobalState *s) : Sampler(s) {}

void Image2D::commit()
{
  // TODO
}

} // namespace barney_device