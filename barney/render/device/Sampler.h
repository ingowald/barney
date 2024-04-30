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

#include "barney/render/device/HitAttributes.h"
#include <cuda.h>

namespace barney {
  namespace render {
    namespace device {

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
    
      struct Sampler {
        typedef enum {
          TRANSFORM,
          IMAGE1D,
          IMAGE2D,
          IMAGE3D
        } Type;

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

#ifdef __CUDA_ARCH__
      inline __device__ float4 Sampler::eval(const HitAttributes &inputs) const
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
}
