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
  Sampler(BarneyGlobalState *s);
  ~Sampler() override;

  static Sampler *createInstance(
      std::string_view subtype, BarneyGlobalState *s);

  virtual BNSampler getBarneySampler(BNModel model, int slot) = 0;

  protected:
  virtual void setBarneyParameters() = 0;
  void setBarneySampler(BNModel model, int slot, const char *subtype);
  void cleanup();

  BNSampler m_bnSampler{nullptr};
};

// Subtypes ///////////////////////////////////////////////////////////////////

struct Image1D : public Sampler
{
  Image1D(BarneyGlobalState *s);
  ~Image1D();
  void commit() override;

  bool isValid() const override;

  BNSampler getBarneySampler(BNModel model, int slot) override;

 private:
  void setBarneyParameters() override;

  helium::IntrusivePtr<helium::Array1D> m_image;
  int m_inAttribute{-1};
  BNTextureAddressMode m_wrapMode{BN_TEXTURE_CLAMP};
  bool m_linearFilter{true};
  math::mat4 m_inTransform{math::identity};
  math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};

  BNSampler m_sampler{nullptr};
};

struct Image2D : public Sampler
{
  Image2D(BarneyGlobalState *s);
  ~Image2D();
  void commit() override;

  bool isValid() const override;

  BNSampler getBarneySampler(BNModel model, int slot) override;

 private:
  void setBarneyParameters() override;

  helium::IntrusivePtr<helium::Array2D> m_image;
  int m_inAttribute{-1};
  BNTextureAddressMode m_wrapMode1{BN_TEXTURE_CLAMP};
  BNTextureAddressMode m_wrapMode2{BN_TEXTURE_CLAMP};
  bool m_linearFilter{true};
  math::mat4 m_inTransform{math::identity};
  math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};

  mutable BNSampler m_sampler{nullptr};
  // mutable BNTexture2D m_texture{nullptr};
};

struct TransformSampler : public Sampler
{
  TransformSampler(BarneyGlobalState *s);
  void commit() override;

  BNSampler getBarneySampler(BNModel model, int slot) override;

 private:
  void setBarneyParameters() override;

  int m_inAttribute{-1};
  math::mat4 m_outTransform{math::identity};
  math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
  mutable BNSampler m_sampler{nullptr};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Sampler *, ANARI_SAMPLER);
