// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/HitAttributes.h"
#include "native/Object.h"
#include "native/common/Data.h"
#include "native/common/math.h"
#include "native/render/TypedRead.h"
#include <stack>

namespace BARNEY_NS {
  namespace native {
    
    struct TextureData;
    struct LDGContext;
    struct SamplerRegistry;
    
    struct AttributeTransform {
      inline __rtc_device vec4f applyTo(const vec4f &in) const;
      
      vec4f mat_x;
      vec4f mat_y;
      vec4f mat_z;
      vec4f mat_w;
      vec4f offset;
    };
    
    struct Sampler : public Object {
      typedef std::shared_ptr<Sampler> SP;
      
      typedef enum {
        INVALID=0,
        TRANSFORM,
        IMAGE1D,
        IMAGE2D,
        IMAGE3D,
        PRIMITIVE,
      } Type;

      struct DD {
#if RTC_DEVICE_CODE
        inline __both__ DD() {}
        inline __both__ DD(const DD &other) {
          type = other.type;
          numChannels = other.numChannels;
          arrayData   = other.arrayData;
          arrayOffset = other.arrayOffset;
          arrayType   = other.arrayType;
          arraySize   = other.arraySize;
          
          texture      = other.texture;
          inAttribute  = other.inAttribute;
          inTransform  = other.inTransform;
          outTransform = other.outTransform;
        }
        inline __rtc_device
        vec4f eval(const HitAttributes &inputs, bool dbg) const;
#endif
        // image only:
        AttributeTransform inTransform;
        rtc::TextureObject texture;
        uint8_t            numChannels;

        // primitive sampler only:
        void              *arrayData;
        int                arrayOffset;
        int                arraySize;
        BNDataType         arrayType;
        
        // all types
        uint8_t type=INVALID;
        uint8_t       inAttribute;
        AttributeTransform outTransform;
      };

      virtual DD getDD(Device *device) = 0;

      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "Sampler<>"; }

      
      static Sampler::SP create(LDGContext *context,
                                const std::string &type);

      Sampler(LDGContext *slotContext);
      virtual ~Sampler();
      
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool set4x4f(const std::string &member, const vec4f *value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      bool setString(const std::string &member,
                     const std::string &value) override;
      bool setObject(const std::string &member,
                   const std::shared_ptr<Object> &value) override;
      /*! @} */
      // ------------------------------------------------------------------

      /*! the registry from whom we got our 'samplerID' - this mainly
          exists for lifetime reasons, to make sure the registry
          doesn't die before we do, because we have to release our
          samplerID when we die */
      const std::shared_ptr<SamplerRegistry> samplerRegistry;
      const int   samplerID;
      int   inAttribute  { ATTRIBUTE_0 };
      mat4f outTransform { mat4f::identity() };
      vec4f outOffset    { 0.f, 0.f, 0.f, 0.f };
      vec4f borderColor  { 0.f, 0.f, 0.f, 0.f };
      DevGroup::SP const devices;
    };

    struct TransformSampler : public Sampler {
      TransformSampler(LDGContext *slotContext)
        : Sampler(slotContext)
      {}
      std::string toString() const override
      { return "TransformSampler"; }
      DD getDD(Device *device) override;
    };

    struct PrimitiveSampler : public Sampler {
      PrimitiveSampler(LDGContext *slotContext);
      ~PrimitiveSampler() override;
      
      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool setData(const std::string &member,
                   const std::shared_ptr<Data> &value) override;
      bool set1i(const std::string &member, const int   &value) override;
      /*! @} */
      // ------------------------------------------------------------------
      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "PrimitiveSampler"; }


      DD getDD(Device *device) override;
      PODData::SP arrayData;
      // format of the entries in the array
      BNDataType arrayType;
      int        arrayOffset = 0;
    };
      
    /*! sampler that operates on rtc-supported texture types; can
        operate on 1D (for ANARI IMAGE1D sampler), 2D (ANARI IMAGE2D)
        and 3D (ANARI IMAGE3D) textures */
    struct TextureSampler : public Sampler {
      TextureSampler(LDGContext *slotContext,
                     int numDims);
      ~TextureSampler() override;

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool setObject(const std::string &member,
                     const std::shared_ptr<Object> &value) override;
      bool set4x4f(const std::string &member, const vec4f *value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      bool set1i(const std::string &member, const int   &value) override;
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------

      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "TextureSampler"+std::to_string(numDims)+"D"; }

      DD getDD(Device *device) override;
      
      struct PLD {
        rtc::Texture *rtcTexture = 0;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;
      
      mat4f inTransform               { mat4f::identity()   };
      vec4f inOffset                  { 0.f, 0.f, 0.f, 0.f  };
      BNTextureAddressMode wrapModes[3]
      = { BN_TEXTURE_WRAP, BN_TEXTURE_WRAP, BN_TEXTURE_WRAP };
      BNTextureFilterMode  filterMode { BN_TEXTURE_LINEAR   };
      BNTextureColorSpace  colorSpace { BN_COLOR_SPACE_LINEAR};
      const int            numDims;
      std::shared_ptr<TextureData> textureData{ 0 };
    };
    
#if RTC_DEVICE_CODE
    inline __rtc_device
    vec4f AttributeTransform::applyTo(const vec4f &in) const
    {
      vec4f out = offset;
      out = out + in.x * mat_x;
      out = out + in.y * mat_y;
      out = out + in.z * mat_z;
      out = out + in.w * mat_w;
      return out;
    }
    
    inline __rtc_device
    vec4f Sampler::DD::eval(const HitAttributes &inputs,
                             bool dbg) const
    {
      if (type == PRIMITIVE) {
        if (!arrayData)
          return vec4f(0.f,0.f,0.f,1.f);

        return typedRead(arrayData, arrayType,
                         arrayOffset+inputs.primID,
                         arraySize);
      }
      vec4f in  = inputs.get((AttributeKind)inAttribute,dbg);
      vec4f out = in;
      if (type == TRANSFORM) {
        return outTransform.applyTo(out);
      }
      
      vec4f coord = inTransform.applyTo(in);
      vec4f fromTex = coord;
      if (type == IMAGE1D) { 
        fromTex = rtc::tex1D<vec4f>(texture,coord.x);
      } else if (type == IMAGE2D) {
        fromTex = rtc::tex2D<vec4f>(texture,coord.x,coord.y);
      } else if (type == IMAGE3D) {
        fromTex = rtc::tex3D<vec4f>(texture,coord.x,coord.y,coord.z);
      } else
        return vec4f(1.f,0.f,0.f,1.f);
      
      // iw - numchannels == 0 can't happen, that's not a valid
      // value
      if (numChannels <= 1) fromTex.y = 0.f;
      if (numChannels <= 2) fromTex.z = 0.f;
      if (numChannels <= 3) fromTex.w = 1.f;
      out = fromTex;
      auto ret = outTransform.applyTo(out);
      return ret;
    }
#endif
    
  } // ::BARNEY_NS::native
} // ::BARNEY_NS
