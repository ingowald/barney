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

namespace barney {
  struct TextureData;
  
  namespace render {

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
      auto print = [&](const char *t, float4 v)
      { printf("  %s %f %f %f %f\n",t,v.x,v.y,v.z,v.w); };
               
      if (dbg) {
        print("mat0 ",mat[0]);
        print("mat1 ",mat[1]);
        print("mat2 ",mat[2]);
        print("mat3 ",mat[3]);
        print("ofs  ",offset);
      }
      vec4f out = load(offset);
      out = out + in.x * load(mat[0]);
      out = out + in.y * load(mat[1]);
      out = out + in.z * load(mat[2]);
      out = out + in.w * load(mat[3]);
      if (dbg) {
        print("applying this to ",in);
        print("      -> gets us ",out);
      }
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
            cudaTextureObject_t texture;
            int                 numChannels;
          } image;
        };
      };

      static Sampler::SP create(ModelSlot *owner, const std::string &type);

      Sampler(ModelSlot *owner);
      virtual ~Sampler();
      
      virtual void createDD(DD &dd, int devID) {};
      virtual void freeDD(DD &dd, int devID) {};

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

      std::vector<DD> perDev;
      
      const int   samplerID;
      int   inAttribute  { render::ATTRIBUTE_0 };
      mat4f outTransform { mat4f::identity() };
      vec4f outOffset    { 0.f, 0.f, 0.f, 0.f };
    };

    struct TransformSampler : public Sampler {
      TransformSampler(ModelSlot *owner)
        : Sampler(owner)
      {}
      void createDD(DD &dd, int devID) override;
      void freeDD(DD &dd, int devID) override;
    };

    struct TextureSampler : public Sampler {
      TextureSampler(ModelSlot *owner, int numDims)
        : Sampler(owner), numDims(numDims)
      { }
      
      virtual ~TextureSampler() = default;

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

      void createDD(DD &dd, int devID) override;
      void freeDD(DD &dd, int devID) override;
      
      mat4f inTransform { mat4f::identity() };
      vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
      BNTextureAddressMode wrapModes[3] = { BN_TEXTURE_WRAP, BN_TEXTURE_WRAP, BN_TEXTURE_WRAP };
      BNTextureFilterMode filterMode = BN_TEXTURE_LINEAR;
      const int   numDims=0;
      std::shared_ptr<TextureData> textureData{ 0 };
    };
  
    

#ifdef __CUDACC__
    inline __device__ float4 Sampler::DD::eval(const HitAttributes &inputs, bool dbg) const
    {
      if (dbg) printf("evaluting sampler %p texture %li\n",this,image.texture);
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
