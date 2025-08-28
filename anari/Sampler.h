// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "common.h"
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"

namespace barney_device {

struct Sampler : public Object
{
  Sampler(BarneyGlobalState *s, const char *barneySubtype);
  ~Sampler() override;

  static Sampler *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  BNSampler getBarneySampler();

 protected:
  BNSampler m_bnSampler{nullptr};

  // this should atually live with the data array that the image
  // sampler(s) are referencing:
  BNTextureData m_bnTextureData{nullptr};
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Image1D : public Sampler
{
  Image1D(BarneyGlobalState *s);
  ~Image1D() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

 private:
  helium::IntrusivePtr<helium::Array1D> m_image;
  std::string m_inAttribute;
  BNTextureAddressMode m_wrapMode{BN_TEXTURE_CLAMP};
  bool m_linearFilter{true};
  math::mat4 m_inTransform{math::identity};
  math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

struct Image2D : public Sampler
{
  Image2D(BarneyGlobalState *s);
  ~Image2D() override;
  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

 private:
  helium::IntrusivePtr<helium::Array2D> m_image;
  std::string m_inAttribute;
  BNTextureAddressMode m_wrapMode1{BN_TEXTURE_CLAMP};
  BNTextureAddressMode m_wrapMode2{BN_TEXTURE_CLAMP};
  bool m_linearFilter{true};
  math::mat4 m_inTransform{math::identity};
  math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

struct TransformSampler : public Sampler
{
  TransformSampler(BarneyGlobalState *s);
  ~TransformSampler() override;
  void commitParameters() override;
  void finalize() override;

 private:
  std::string m_inAttribute;
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Sampler *, ANARI_SAMPLER);
