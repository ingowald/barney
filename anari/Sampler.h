// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Array.h"
#include "Object.h"

namespace barney_device {

struct Sampler : public Object
{
  Sampler(BarneyGlobalState *s);
  ~Sampler() override;

  static Sampler *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  BNTexture2D barneyTexture() const;

 protected:
  BNTexture2D m_barneyTexture{nullptr};
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Image1D : public Sampler
{
  Image1D(BarneyGlobalState *s);
  void commit() override;

 private:
  helium::IntrusivePtr<Array1D> m_image;
  std::string m_inAttribute{"attribute0"};
  bool m_linearFilter{true};
  std::string m_wrapMode{"clampToEdge"};
  math::mat4 m_inTransform{math::mat4(linalg::identity)};
  math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  math::mat4 m_outTransform{math::mat4(linalg::identity)};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

struct Image2D : public Sampler
{
  Image2D(BarneyGlobalState *s);
  void commit() override;
};

} // namespace barney_device
