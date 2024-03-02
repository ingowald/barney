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

#include "barney/material/device/DG.h"

namespace barney {
  namespace render {
    
    struct MiniMaterial {
      struct HitBSDF {
        inline __device__ vec3f eval(DG dg, vec3f w_i, bool dbg) const;
        inline __device__
        vec3f getAlbedo(bool dbg) const
        {
          return vec3f(baseColor);
        }

        vec3h baseColor;
        half  ior;
        half  metallic;
        half  transmission;
      };
      struct DD {
  
        inline __device__
        float getAlpha(vec2f tc, bool isShadowRay) const
        {
#ifdef __CUDACC__
          if (alphaTexture)
            return tex2D<float4>(alphaTexture,tc.x,tc.y).w;
          if (colorTexture)
            return tex2D<float4>(colorTexture,tc.x,tc.y).w;
#endif
          if (isShadowRay)
            return 1.f - transmission;
          return 1.f;
        }
        inline __device__
        bool hasAlpha(bool isShadowRay) const
        {
          return colorTexture || alphaTexture
            || (isShadowRay && transmission > 0.f);
        }
  
        inline __device__
        void make(HitBSDF &hit, 
                  vec2f tc,
                  vec3f geometryColor,
                  bool dbg) const
        {
#ifdef __CUDACC__
          hit.baseColor = this->baseColor;
          hit.ior = this->ior;
          hit.transmission = this->transmission;
          if (!isnan(geometryColor.x)) {
            hit.baseColor = geometryColor;
          } else if (this->colorTexture) {
            float4 fromTex = tex2D<float4>(this->colorTexture,tc.x,tc.y);
            hit.baseColor = (vec3f&)fromTex;
          } else {
            hit.baseColor = this->baseColor;
          }
#endif
        }
        
        vec3f baseColor;
        float ior;
        float transmission;
        // float roughness;
        float metallic;
        cudaTextureObject_t colorTexture;
        cudaTextureObject_t alphaTexture;
      };
    };

  }
}
