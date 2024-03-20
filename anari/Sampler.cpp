// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"

namespace barney_device {

Sampler::Sampler(BarneyGlobalState *s) : Object(ANARI_SAMPLER, s) {}

Sampler::~Sampler() = default;

Sampler *Sampler::createInstance(std::string_view subtype, BarneyGlobalState *s)
{
  if (subtype == "transform")
    return new TransformSampler(s);
  else
    return (Sampler *)new UnknownObject(ANARI_SAMPLER, s);
}

// Subtypes ///////////////////////////////////////////////////////////////////

// TransformSampler //

TransformSampler::TransformSampler(BarneyGlobalState *s) : Sampler(s) {}

inline int toAttribute(std::string str) {
  if (str == "attribute0")
    return 0;
  else if (str == "attribute1")
    return 1;
  else if (str == "attribute2")
    return 2;
  else if (str == "attribute3")
    return 3;
  else if (str == "color")
    return 4;
  else if (str == "none")
    return -1;
  return -1;
}

void TransformSampler::commit()
{
  Object::commit();

  m_inAttribute =
      toAttribute(getParamString("inAttribute", "attribute0"));
  //m_outTransform = getParam<math::mat4>("outTransform", math::identity);
  m_outTransform = math::identity;
  getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
  getParam("transform", ANARI_FLOAT32_MAT4, &m_outTransform);
  m_outOffset = getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
