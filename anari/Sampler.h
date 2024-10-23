// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "common.h"
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"

namespace tally_device {

  struct Sampler : public Object
  {
    Sampler(TallyGlobalState *s);
    ~Sampler() override;

    static Sampler *createInstance(
                                   std::string_view subtype, TallyGlobalState *s);

    TallySampler::SP getTallySampler(TallyModel::SP model, int slot);
    // virtual TallySampler::SP getTallySampler(TallyModel::SP model, int slot) = 0;

  protected:
    virtual void createTallySampler(TallyModel::SP model, int slot) = 0;
    // void setTallySampler(TallyModel::SP model, int slot, const char *subtype);
    void cleanup();

    mutable TallySampler::SP m_bnSampler{nullptr};
  
    // this should atually live with the data array that the image
    // sampler(s) are referencing:
    mutable BNTextureData m_bnTextureData{nullptr};
  };

  // Subtypes ///////////////////////////////////////////////////////////////////

  struct Image1D : public Sampler
  {
    Image1D(TallyGlobalState *s);
    ~Image1D();
    void commit() override;

    bool isValid() const override;

    // TallySampler::SP getTallySampler(TallyModel::SP model, int slot) override;

  private:
    void createTallySampler(TallyModel::SP model, int slot) override;

    helium::IntrusivePtr<helium::Array1D> m_image;
    std::string m_inAttribute;
    BNTextureAddressMode m_wrapMode{BN_TEXTURE_CLAMP};
    bool m_linearFilter{true};
    math::mat4 m_inTransform{math::identity};
    math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
    math::mat4 m_outTransform{math::identity};
    math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};

    // TallySampler::SP m_bnSampler{nullptr};
  };

  struct Image2D : public Sampler
  {
    Image2D(TallyGlobalState *s);
    ~Image2D();
    void commit() override;

    bool isValid() const override;

    // TallySampler::SP getTallySampler(TallyModel::SP model, int slot) override;

  private:
    void createTallySampler(TallyModel::SP model, int slot) override;

    helium::IntrusivePtr<helium::Array2D> m_image;
    std::string          m_inAttribute;
#if TALLY
    BNTextureAddressMode m_wrapMode1{BN_TEXTURE_CLAMP};
    BNTextureAddressMode m_wrapMode2{BN_TEXTURE_CLAMP};
#endif
    bool m_linearFilter{true};
    math::mat4 m_inTransform{math::identity};
    math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
    math::mat4 m_outTransform{math::identity};
    math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};

    // mutable TallySampler::SP m_bnSampler{nullptr};
    // mutable BNTexture2D m_texture{nullptr};
  };

  struct TransformSampler : public Sampler
  {
    TransformSampler(TallyGlobalState *s);
    void commit() override;

    // TallySampler::SP getTallySampler(TallyModel::SP model, int slot) override;

  private:
    void createTallySampler(TallyModel::SP model, int slot) override;

    std::string  m_inAttribute;
    math::mat4   m_outTransform{math::identity};
    math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
    // mutable TallySampler::SP m_bnSampler{nullptr};
  };

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Sampler *, ANARI_SAMPLER);
