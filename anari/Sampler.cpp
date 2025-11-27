// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Sampler.h"
// std
#include <cassert>
#include <iostream>
#include "Device.h"
#if BARNEY_MPI
#include <mpi.h>
#endif

#include "Array.h"
#include "Frame.h"
// std
#include <cstring>

#include "generated/anari_library_barney_queries.h"


namespace barney_device {

  // Helper functions /////////////////////////////////////////////////////////

  BNTextureData makeBarneyTextureData(Object *objectToReportErrorWith,
                                      BarneyGlobalState *state,
                                      helium::Array *input,
                                      int width,
                                      int height,
                                      int depth=-1)
  {
    int slot = state->slot;
    auto context = state->tether->context;

    if (depth == -1) {
      // this is 2D data
      if (input->elementType() == ANARI_FLOAT32_VEC4) {
        return bnTextureData2DCreate(context,
                                     slot,
                                     BN_FLOAT4,
                                     width,
                                     height,
                                     input->dataAs<anari::math::float4>());
      } else if (input->elementType() == ANARI_FLOAT32) {
        return bnTextureData2DCreate(context,
                                     slot,
                                     BN_FLOAT,
                                     width,
                                     height,
                                     input->dataAs<float>());
      } else if (input->elementType() == ANARI_UFIXED8_VEC4) {
        return bnTextureData2DCreate(context,
                                     slot,
                                     BN_UFIXED8_RGBA,
                                     width,
                                     height,
                                     input->data());
      } else {
        std::vector<uint32_t> texels;
        objectToReportErrorWith->reportMessage
          (ANARI_SEVERITY_PERFORMANCE_WARNING,
           "banari doing 2D texture format conversion (%s)",
           anari::toString(input->elementType()));
        if (!convert_to_rgba8(input, texels)) {
          objectToReportErrorWith->reportMessage
            (ANARI_SEVERITY_ERROR,
             "unsupported texel type (%s) in 2D texture format conversion",
            anari::toString(input->elementType()));
          return {};
        }
        return bnTextureData2DCreate(context, slot, BN_UFIXED8_RGBA,
                                     width, height, texels.data());
      }
    } else {
      // this is 3D data
      if (input->elementType() == ANARI_FLOAT32) {
        return bnTextureData3DCreate(context,
                                     slot,
                                     BN_FLOAT,
                                     width,
                                     height,
                                     depth,
                                     input->dataAs<float>());
      } else if (input->elementType() == ANARI_UFIXED8) {
        return bnTextureData3DCreate(context,
                                     slot,
                                     BN_UFIXED8,
                                     width,
                                     height,
                                     depth,
                                     input->data());
      } else if (input->elementType() == ANARI_INT8) {
        return bnTextureData3DCreate(context,
                                     slot,
                                     BN_UINT8,
                                     width,
                                     height,
                                     depth,
                                     input->data());
      } else if (input->elementType() == ANARI_UFIXED16) {
        return bnTextureData3DCreate(context,
                                     slot,
                                     BN_UFIXED16,
                                     width,
                                     height,
                                     depth,
                                     input->data());
      } else if (input->elementType() == ANARI_FLOAT32_VEC4) {
        return bnTextureData3DCreate(context,
                                     slot,
                                     BN_FLOAT32_VEC4,
                                     width,
                                     height,
                                     depth,
                                     input->dataAs<anari::math::float4>());
      } else {
        std::vector<uint32_t> texels;
        objectToReportErrorWith->reportMessage
          (ANARI_SEVERITY_PERFORMANCE_WARNING,
           "banari doing 3D texture format conversion (%s)",
           anari::toString(input->elementType()));
        if (!convert_to_rgba8(input, texels)) {
          objectToReportErrorWith->reportMessage
            (ANARI_SEVERITY_ERROR,
             "unsupported texel type (%s) in 3D texture format conversion",
            anari::toString(input->elementType()));
          return {};
        }
        return bnTextureData2DCreate(context, slot, BN_UFIXED8_RGBA,
                                     width, height, texels.data());
      }
    }
  }

  // Sampler definitions ////////////////////////////////////////////////////////

  Sampler::Sampler(BarneyGlobalState *s, const char *barneySubtype)
    : Object(ANARI_SAMPLER, s)
  {
    int slot = deviceState()->slot;
    auto context = deviceState()->tether->context;
    m_bnSampler = bnSamplerCreate(context, slot, barneySubtype);
  }

  Sampler::~Sampler()
  {
    BANARI_TRACK_LEAKS(std::cout << "#banari: ~Sampler deconstructing" << std::endl);
    bnRelease(m_bnSampler);
    if (m_bnTextureData) {
      bnRelease(m_bnTextureData);
      m_bnTextureData = nullptr;
    }
  }

  Sampler *Sampler::createInstance(std::string_view subtype, BarneyGlobalState *s)
  {
    if (subtype == "image1D")
      return new Image1D(s);
    else if (subtype == "image2D")
      return new Image2D(s);
    else if (subtype == "image3D")
      return new Image3D(s);
    else if (subtype == "transform")
      return new TransformSampler(s);
    else
      return (Sampler *)new UnknownObject(ANARI_SAMPLER, subtype, s);
  }

  BNSampler Sampler::getBarneySampler()
  {
    if (!isValid())
      return {};
    return m_bnSampler;
  }

  // Subtypes ///////////////////////////////////////////////////////////////////

  // Image1D //

  Image1D::Image1D(BarneyGlobalState *s) : Sampler(s, "texture2D") {}

  Image1D::~Image1D() = default;

  void Image1D::commitParameters()
  {
    Sampler::commitParameters();
    m_image = getParamObject<helium::Array1D>("image");
    m_inAttribute = getParamString("inAttribute", "attribute0");
    m_linearFilter = getParamString("filter", "linear") != "nearest";
    m_wrapMode = toBarneyAddressMode(getParamString("wrapMode", "clampToEdge"));
    m_inTransform = math::identity;
    getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    m_inOffset =
      getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    m_outTransform = math::identity;
    getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  }

  bool Image1D::isValid() const
  {
    return m_image;
  }

  void Image1D::finalize()
  {
    if (m_bnTextureData) {
      bnRelease(m_bnTextureData);
      m_bnTextureData = 0;
    }
    // if (m_bnSampler) {
    //   bnRelease(m_bnSampler);
    //   m_bnSampler = 0;
    // }
    if (!m_image) {
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "Image1D::finalize() without a valid 'image' parameter");
      return;
    }

    m_bnTextureData
      = makeBarneyTextureData(this,
                              deviceState(), m_image.ptr,
                              (int)m_image->size(), 1);
    // ------------------------------------------------------------------
    // now, create sampler over those texels
    // ------------------------------------------------------------------

    BNTextureFilterMode filterMode =
      m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST;

    bnSet1i(m_bnSampler, "filterMode", (int)filterMode);
    bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode);
    bnSet1i(m_bnSampler, "wrapMode1", (int)BN_TEXTURE_CLAMP);
    bnSet4x4fv(m_bnSampler, "inTransform", (const bn_float4 *)&m_inTransform);
    bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
    bnSet4f(m_bnSampler,
            "inOffset",
            m_inOffset.x,
            m_inOffset.y,
            m_inOffset.z,
            m_inOffset.w);
    bnSet4f(m_bnSampler,
            "outOffset",
            m_outOffset.x,
            m_outOffset.y,
            m_outOffset.z,
            m_outOffset.w);
    bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
    bnSetObject(m_bnSampler, "textureData", m_bnTextureData);
    bnCommit(m_bnSampler);
  }

  // Image2D //

  Image2D::Image2D(BarneyGlobalState *s) : Sampler(s, "texture2D") {}

  Image2D::~Image2D() = default;

  void Image2D::commitParameters()
  {
    Sampler::commitParameters();
    m_image = getParamObject<helium::Array2D>("image");
    m_inAttribute = getParamString("inAttribute", "attribute0");
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
  }

  void Image2D::finalize()
  {
    if (m_bnTextureData) {
      bnRelease(m_bnTextureData);
      m_bnTextureData = 0;
    }
    if (!m_image) {
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "Image2D::finalize() without a valid 'image' parameter");
      return;
    }

    m_bnTextureData
      = makeBarneyTextureData(this,
                              deviceState(), m_image.ptr,
                              m_image->size().x, m_image->size().y);

    BNTextureFilterMode filterMode =
      m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST;
    bnSet1i(m_bnSampler, "filterMode", (int)filterMode);
    bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode1);
    bnSet1i(m_bnSampler, "wrapMode1", (int)m_wrapMode2);
    bnSet4x4fv(m_bnSampler, "inTransform", (const bn_float4 *)&m_inTransform);
    bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
    bnSet4f(m_bnSampler,
            "inOffset",
            m_inOffset.x,
            m_inOffset.y,
            m_inOffset.z,
            m_inOffset.w);
    bnSet4f(m_bnSampler,
            "outOffset",
            m_outOffset.x,
            m_outOffset.y,
            m_outOffset.z,
            m_outOffset.w);
    bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
    bnSetObject(m_bnSampler, "textureData", m_bnTextureData);
    bnCommit(m_bnSampler);
  }

  bool Image2D::isValid() const
  {
    return m_image;
  }

  // Image3D //

  Image3D::Image3D(BarneyGlobalState *s) : Sampler(s, "texture3D") {}

  Image3D::~Image3D() = default;

  void Image3D::commitParameters()
  {
    Sampler::commitParameters();
    m_image = getParamObject<helium::Array3D>("image");
    m_inAttribute = getParamString("inAttribute", "attribute0");
    m_linearFilter = getParamString("filter", "linear") != "nearest";
    m_wrapMode1 = toBarneyAddressMode(getParamString("wrapMode1", "clampToEdge"));
    m_wrapMode2 = toBarneyAddressMode(getParamString("wrapMode2", "clampToEdge"));
    m_wrapMode3 = toBarneyAddressMode(getParamString("wrapMode3", "clampToEdge"));
    m_inTransform = math::identity;
    getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    m_inOffset =
      getParam<math::float4>("inOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    m_outTransform = math::identity;
    getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
  }

  void Image3D::finalize()
  {
    if (m_bnTextureData) {
      bnRelease(m_bnTextureData);
      m_bnTextureData = 0;
    }
    // if (m_bnSampler) {
    //   bnRelease(m_bnSampler);
    //   m_bnSampler = 0;
    // }
    if (!m_image) {
      reportMessage(ANARI_SEVERITY_DEBUG,
                    "Image3D::finalize() without a valid 'image' parameter");
      return;
    }
    m_bnTextureData = makeBarneyTextureData(this,
                                            deviceState(), m_image.ptr,
                                            m_image->size().x,
                                            m_image->size().y,
                                            m_image->size().z);

    BNTextureFilterMode filterMode =
      m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST;
    bnSet1i(m_bnSampler, "filterMode", (int)filterMode);
    bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode1);
    bnSet1i(m_bnSampler, "wrapMode1", (int)m_wrapMode2);
    bnSet1i(m_bnSampler, "wrapMode2", (int)m_wrapMode3);
    bnSet4x4fv(m_bnSampler, "inTransform", (const bn_float4 *)&m_inTransform);
    bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
    bnSet4f(m_bnSampler,
            "inOffset",
            m_inOffset.x,
            m_inOffset.y,
            m_inOffset.z,
            m_inOffset.w);
    bnSet4f(m_bnSampler,
            "outOffset",
            m_outOffset.x,
            m_outOffset.y,
            m_outOffset.z,
            m_outOffset.w);
    bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
    bnSetObject(m_bnSampler, "textureData", m_bnTextureData);
    bnCommit(m_bnSampler);
  }

  bool Image3D::isValid() const
  {
    return m_image;
  }

  /// Transform ///

  TransformSampler::TransformSampler(BarneyGlobalState *s)
    : Sampler(s, "transform")
  {}

  TransformSampler::~TransformSampler() = default;

  void TransformSampler::commitParameters()
  {
    Sampler::commitParameters();
    m_outTransform = math::identity;
    getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    m_outOffset =
      getParam<math::float4>("outOffset", math::float4(0.f, 0.f, 0.f, 0.f));
    m_inAttribute = getParamString("inAttribute", "attribute0");
  }

  void TransformSampler::finalize()
  {
    bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
    bnSet4f(m_bnSampler,
            "outOffset",
            m_outOffset.x,
            m_outOffset.y,
            m_outOffset.z,
            m_outOffset.w);
    bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
    bnCommit(m_bnSampler);
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Sampler *);
