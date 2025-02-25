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
#include "rtcore/Frontend.h"
#include <stack>

namespace BARNEY_NS {
  struct TextureData;
  struct SlotContext;

  namespace render {
    struct SamplerRegistry;
    
    struct AttributeTransform {
      inline __rtc_device vec4f applyTo(vec4f in, bool dbg) const;
      
      rtc::float4 mat[4];
      rtc::float4 offset;
    };
    
    struct Sampler : public barney_api::Sampler {
      typedef std::shared_ptr<Sampler> SP;
      
      typedef enum {
        TRANSFORM,
        IMAGE1D,
        IMAGE2D,
        IMAGE3D,
        INVALID=-1
      } Type;

      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device DD() {}
        inline __rtc_device DD(DD &&other) { memcpy(this,&other,sizeof(other)); }
        inline __rtc_device
        vec4f eval(const HitAttributes &inputs, bool dbg) const;
#endif
        Type type=INVALID;
        AttributeKind      inAttribute;
        AttributeTransform outTransform;
        union {
          struct {
            AttributeTransform         inTransform;
            rtc::device::TextureObject texture;
            int                        numChannels;
          } image;
        };
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
    vec4f AttributeTransform::applyTo(vec4f in,
                                       bool dbg) const
    {
      vec4f out = rtc::load(offset);
      out = out + in.x * rtc::load(mat[0]);
      out = out + in.y * rtc::load(mat[1]);
      out = out + in.z * rtc::load(mat[2]);
      out = out + in.w * rtc::load(mat[3]);
      return (const vec4f&)out;
    }
    
    inline __rtc_device
    vec4f Sampler::DD::eval(const HitAttributes &inputs,
                             bool dbg) const
    {
      // dbg = true;
      if (dbg) printf("evaluting sampler %p texture %p\n",this,
                      (void*)image.texture);
      vec4f in  = inputs.get(inAttribute);
      if (dbg) printf("in is %f %f %f %f\n",in.x,in.y,in.z,in.w);
      if (type != TRANSFORM) {
        vec4f coord = image.inTransform.applyTo(in,dbg);
        if (dbg) printf("coord is %f %f %f %f\n",coord.x,coord.y,coord.z,coord.w);
        vec4f fromTex;
        if (type == IMAGE1D)
          fromTex = rtc::tex1D<vec4f>(image.texture,coord.x);
        else if (type == IMAGE2D) {
          if (dbg) printf("sampling 2d texture %p at %f %f\n",
                          (int*)image.texture,coord.x,coord.y);
          fromTex = rtc::tex2D<vec4f>(image.texture,coord.x,coord.y);
        } else
          fromTex = rtc::tex3D<vec4f>(image.texture,coord.x,coord.y,coord.z);

        if (dbg) printf("fromTex is %f %f %f %f\n",fromTex.x,fromTex.y,fromTex.z,fromTex.w);
        in.x = fromTex.x;

        if (image.numChannels >= 1) in.y = fromTex.y;
        if (image.numChannels >= 2) in.z = fromTex.z;
        if (image.numChannels >= 3) in.w = fromTex.w;
        if (dbg) printf("numchan %i -> %f %f %f %f\n",
                        image.numChannels,
                        in.x,in.y,in.z,in.w);
      }
      vec4f out = outTransform.applyTo(in,dbg);
      return out;
    }
#endif
  }
}
