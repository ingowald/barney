// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once 

#include "Object.h"
#include "common.h"
#include "helium/array/Array1D.h"

namespace barney_device {

struct Sampler : public Object
{
  Sampler(BarneyGlobalState *s);
  ~Sampler() override;

  static Sampler *createInstance(
      std::string_view subtype, BarneyGlobalState *s);
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Image1D : public Sampler
{
  Image1D(BarneyGlobalState *s);
  void commit() override;

  helium::IntrusivePtr<helium::Array1D> m_image;
  int m_inAttribute{-1};
  WrapMode m_wrapMode{Clamp};
  bool m_linearFilter{true};
  math::mat4 m_inTransform{math::identity};
  math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

struct TransformSampler : public Sampler
{
  TransformSampler(BarneyGlobalState *s);
  void commit() override;

  int m_inAttribute{-1};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Sampler *, ANARI_SAMPLER);
