// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "barney/render/HitAttributes.h"
#include "barney/Object.h"
#include "barney/common/mat4.h"
#include "barney/common/math.h"
#include <stack>
#if RTC_DEVICE_CODE
# include "rtcore/ComputeInterface.h"
#endif

namespace BARNEY_NS {
  struct TextureData;
  struct SlotContext;

  namespace render {
    struct SamplerRegistry;
    
    struct AttributeTransform {
      inline __rtc_device vec4f applyTo(const vec4f &in) const;
      
      // rtc::float4 mat[4];
      // rtc::float4 offset;
      vec4f mat_x;
      vec4f mat_y;
      vec4f mat_z;
      vec4f mat_w;
      vec4f offset;
    };
    
    struct Sampler : public barney_api::Sampler {
      typedef std::shared_ptr<Sampler> SP;
      
      typedef enum {
        INVALID=0,
        TRANSFORM,
        IMAGE1D,
        IMAGE2D,
        IMAGE3D,
      } Type;

      struct DD {
#if RTC_DEVICE_CODE
        inline __both__ DD() {}
        inline __both__ DD(const DD &other) {
          type = other.type;
          numChannels = other.numChannels;
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
        
        // all types
        uint8_t type=INVALID;
        uint8_t       inAttribute;
        AttributeTransform outTransform;
      };

      virtual DD getDD(Device *device) = 0;
      
      static Sampler::SP create(SlotContext *context,
                                const std::string &type);

      Sampler(SlotContext *slotContext);
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
      int   inAttribute  { render::ATTRIBUTE_0 };
      mat4f outTransform { mat4f::identity() };
      vec4f outOffset    { 0.f, 0.f, 0.f, 0.f };
      DevGroup::SP const devices;
    };

    struct TransformSampler : public Sampler {
      TransformSampler(SlotContext *slotContext)
        : Sampler(slotContext)
      {}
      DD getDD(Device *device) override;
      // void createDD(DD &dd, int devID) override;
      // void freeDD(DD &dd, int devID) override;
    };

    struct TextureSampler : public Sampler {
      TextureSampler(SlotContext *slotContext,
                     int numDims);
      virtual ~TextureSampler();

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
      
      mat4f inTransform { mat4f::identity() };
      vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
      BNTextureAddressMode wrapModes[3]
      = { BN_TEXTURE_WRAP, BN_TEXTURE_WRAP, BN_TEXTURE_WRAP };
      BNTextureFilterMode filterMode = BN_TEXTURE_LINEAR;
      const int   numDims=0;
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
      vec4f in  = inputs.get((AttributeKind)inAttribute);
      if (type != TRANSFORM) {
        vec4f coord = inTransform.applyTo(in);
        if (type == IMAGE1D) {
          vec4f fromTex = rtc::tex1D<vec4f>(texture,coord.x);
          if (numChannels == 1) {
            fromTex.y = fromTex.z = 0.f; fromTex.w = 1.f;
          } else if (numChannels == 2) {
            fromTex.z = 0.f; fromTex.w = 1.f;
          } else if (numChannels == 3) {
            fromTex.w = 1.f;
          }
          return outTransform.applyTo(fromTex);
        }
        if (type == IMAGE2D) {
          vec4f fromTex = rtc::tex2D<vec4f>(texture,coord.x,coord.y);
          if (numChannels == 1) {
            fromTex.y = fromTex.z = 0.f; fromTex.w = 1.f;
          } else if (numChannels == 2) {
            fromTex.z = 0.f; fromTex.w = 1.f;
          } else if (numChannels == 3) {
            fromTex.w = 1.f;
          }
          return outTransform.applyTo(fromTex);
        }
        return coord;
      }
      vec4f out = outTransform.applyTo(in);
      return out;
    }
#endif
  }
}
