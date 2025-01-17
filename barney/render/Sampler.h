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
#include <cuda.h>
#include <stack>

namespace barney {
  struct TextureData;
  
  namespace render {
    struct SamplerRegistry;
    
    struct AttributeTransform {
#ifdef __CUDACC__
      inline __device__ float4 applyTo(float4 in, bool dbg) const;
#endif
      float4 mat[4];
      float4 offset;
    };
      
#ifdef __CUDACC__
    inline __device__ vec4f load(float4 v) { return (const vec4f&)v; }
    
    inline __device__ float4 AttributeTransform::applyTo(float4 in, bool dbg) const
    {
      vec4f out = load(offset);
      out = out + in.x * load(mat[0]);
      out = out + in.y * load(mat[1]);
      out = out + in.z * load(mat[2]);
      out = out + in.w * load(mat[3]);
      return (const float4&)out;
    }
#endif
    
    struct Sampler : public SlottedObject {
      typedef std::shared_ptr<Sampler> SP;
      
      typedef enum {
        TRANSFORM,
        IMAGE1D,
        IMAGE2D,
        IMAGE3D,
        INVALID=-1
      } Type;

      struct DD {
#ifdef __CUDACC__
        inline __device__ float4 eval(const HitAttributes &inputs, bool dbg) const;
#endif
        
        Type type=INVALID;
        AttributeKind      inAttribute;
        AttributeTransform outTransform;
        union {
          struct {
            AttributeTransform  inTransform;
            // cudaTextureObject_t texture;
            rtc::device::TextureObject texture;
            int                 numChannels;
          } image;
        };
      };

      virtual DD getDD(rtc::Device *device) const = 0;
      
      static Sampler::SP create(Context *context,
                                int slot,
                                const std::string &type);

      Sampler(Context *context, int slot);
      virtual ~Sampler();
      
      // virtual void createDD(DD &dd, int devID) {};
      // virtual void freeDD(DD &dd, int devID) {};

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool setObject(const std::string &member,
                   const std::shared_ptr<Object> &value) override;
      bool set4x4f(const std::string &member, const mat4f &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      bool setString(const std::string &member,
                     const std::string &value) override;
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------

      // std::vector<DD> perDev;
      rtc::Texture *rtcTexture = 0;
      
      /*! the registry from whom we got our 'samplerID' - this mainly
          exists for lifetime reasons, to make sure the registry
          doesn't die before we do, because we have to release our
          samplerID when we die */
      std::shared_ptr<SamplerRegistry> samplerRegistry;
      
      const int   samplerID;
      int   inAttribute  { render::ATTRIBUTE_0 };
      mat4f outTransform { mat4f::identity() };
      vec4f outOffset    { 0.f, 0.f, 0.f, 0.f };
    };

    struct TransformSampler : public Sampler {
      TransformSampler(Context *context, int slot)
        : Sampler(context,slot)
      {}
      DD getDD(rtc::Device *device) const override;
      // void createDD(DD &dd, int devID) override;
      // void freeDD(DD &dd, int devID) override;
    };

    struct TextureSampler : public Sampler {
      TextureSampler(Context *context, int slot, int numDims)
        : Sampler(context,slot), numDims(numDims)
      { }
      
      virtual ~TextureSampler();

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool setObject(const std::string &member,
                   const std::shared_ptr<Object> &value) override;
      bool set4x4f(const std::string &member, const mat4f &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      bool set1i(const std::string &member, const int   &value) override;
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------

      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "TextureSampler"+std::to_string(numDims)+"D"; }

      DD getDD(rtc::Device *device) const override;
      // void createDD(DD &dd, int devID) override;
      // void freeDD(DD &dd, int devID) override;
      
      mat4f inTransform { mat4f::identity() };
      vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
      BNTextureAddressMode wrapModes[3]
      = { BN_TEXTURE_WRAP, BN_TEXTURE_WRAP, BN_TEXTURE_WRAP };
      BNTextureFilterMode filterMode = BN_TEXTURE_LINEAR;
      const int   numDims=0;
      std::shared_ptr<TextureData> textureData{ 0 };
    };
  
    

#ifdef __CUDACC__
    inline __device__ float4 Sampler::DD::eval(const HitAttributes &inputs, bool dbg) const
    {
      dbg = false;
      if (dbg) printf("evaluting sampler %p texture %p\n",this,
                      (void*)image.texture);
      float4 in  = inputs.get(inAttribute);
      if (dbg) printf("in is %f %f %f %f\n",in.x,in.y,in.z,in.w);
      if (type != TRANSFORM) {
        float4 coord = image.inTransform.applyTo(in,dbg);
        if (dbg) printf("coord is %f %f %f %f\n",coord.x,coord.y,coord.z,coord.w);
        float4 fromTex;
        if (type == IMAGE1D)
          fromTex = tex1D<float4>(image.texture,coord.x);
        else if (type == IMAGE2D) {
          if (dbg) printf("sampling 2d texture at %f %f\n",coord.x,coord.y);
          fromTex = tex2D<float4>(image.texture,coord.x,coord.y);
        } else
          fromTex = tex3D<float4>(image.texture,coord.x,coord.y,coord.z);

        if (dbg) printf("fromTex is %f %f %f %f\n",fromTex.x,fromTex.y,fromTex.z,fromTex.w);
        in.x = fromTex.x;
        if (image.numChannels >= 1) in.y = fromTex.y;
        if (image.numChannels >= 2) in.z = fromTex.z;
        if (image.numChannels >= 3) in.w = fromTex.w;
      }
      return outTransform.applyTo(in,dbg);
    }
#endif
  }
}
