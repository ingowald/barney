// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once 

#include "Object.h"

namespace barney_device {

struct Sampler : public Object
{
  Sampler(BarneyGlobalState *s);
  ~Sampler() override;

  static Sampler *createInstance(
      std::string_view subtype, BarneyGlobalState *s);
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct TransformSampler : public Sampler
{
  TransformSampler(BarneyGlobalState *s);
  void commit() override;

  int m_inAttribute{-1};
  math::mat4 m_outTransform;
  math::float4 m_outOffset;
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Sampler *, ANARI_SAMPLER);
