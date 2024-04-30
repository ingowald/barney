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
#include <cuda.h>

namespace barney {
  namespace render {

    struct AttributeTransform {
      inline __device__ float4 applyTo(float4 in) const;
      float4 mat[4];
      float4 offset;
    };
      
    inline __device__ vec4f load(float4 v) { return (const vec4f&)v; }
    
    inline __device__ float4 AttributeTransform::applyTo(float4 in) const
    {
      vec4f out = load(offset);
      out = out + in.x * load(mat[0]);
      out = out + in.y * load(mat[1]);
      out = out + in.z * load(mat[2]);
      out = out + in.w * load(mat[3]);
      return (const float4&)out;
    }
    
    struct Sampler : public SlottedObject {
      typedef std::shared_ptr<Sampler> SP;
      
      typedef enum {
        TRANSFORM,
        IMAGE1D,
        IMAGE2D,
        IMAGE3D
      } Type;

      struct DD {
        inline __device__ float4 eval(const HitAttributes &inputs) const;
        
        Type type;
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
      
      virtual void create(DD &dd, int devID) = 0;
      
      int   samplerID = -1;
      int   inAttribute { render::ATTRIBUTE_0 };
      mat4f outTransform { mat4f::identity() };
      vec4f outOffset { 0.f, 0.f, 0.f, 0.f };
    };

    struct TransformSampler : public Sampler {
      void create(DD &dd, int devID) override;
    };
    struct ImageSampler : public Sampler {
      void create(DD &dd, int devID) override;
    
      mat4f inTransform { mat4f::identity() };
      vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
      Texture::SP image{ 0 };
    };
  
    

#ifdef __CUDA_ARCH__
    inline __device__ float4 Sampler::DD::eval(const HitAttributes &inputs) const
    {
      float4 in  = inputs.get(inAttribute);
      if (type != TRANSFORM) {
        float4 coord = image.inTransform.applyTo(in);
        float4 fromTex;
        if (type == IMAGE1D)
          fromTex = tex1D<float4>(image.texture,coord.x);
        else if (type == IMAGE2D)
          fromTex = tex2D<float4>(image.texture,coord.x,coord.y);
        else
          fromTex = tex3D<float4>(image.texture,coord.x,coord.y,coord.z);

        in.x = fromTex.x;
        if (image.numChannels >= 1) in.y = fromTex.y;
        if (image.numChannels >= 2) in.z = fromTex.z;
        if (image.numChannels >= 3) in.w = fromTex.w;
      }
      return outTransform.applyTo(in);
    }
#endif
  }
}
