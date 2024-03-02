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
#include "barney/material/device/Velvet.h"
#include "barney/material/device/Mini.h"

namespace barney {
  namespace render {
    
    typedef enum { MISS=0, MINI, VELVET, ANARI_PHYSICAL } MaterialType;
  
    /*! device-side implementation of anari "physical" material */
    struct AnariPhysical {
      struct BRDF {
        /*! "BRDFs" are the thing that goes onto a ray, and is used for
          sampling, eval, etc */
        // vec3h reflectance;
      };
      /*! "DDs" are the device-data that gets stored in the associated
        geometry's SBT entry */
      struct DD {
        vec3f baseColor;
      };
    }; // ::barney::render::AnariPhysical    
    
    struct HitBRDF {
      /*! helper function to set this to a matte material, primarily
          for volume data */
      inline __device__ void setMatte(vec3f albedo, vec3f P, vec3f N);
      /*! modulate given BRDF with a color form texture, or colors[] array, etc */
      // inline __device__ void modulateBaseColor(vec3f rbga);
      inline __device__ void setDG(vec3f P, vec3f N, bool dbg=false);
      inline __device__ vec3f getAlbedo(bool dbg=false) const;
      inline __device__ vec3f getN() const;
      inline __device__ vec3f eval(render::DG dg, vec3f w_i, bool dbg=false) const;
      union {
        float3 missColor;
        render::AnariPhysical::BRDF anari;
        render::MiniMaterial::BRDF  mini;
      };
      vec3f P;
      
      struct {
        uint32_t quantized_nx_bits:7;
        uint32_t quantized_nx_sign:1;
        uint32_t quantized_ny_bits:7;
        uint32_t quantized_ny_sign:1;
        uint32_t quantized_nz_bits:7;
        uint32_t quantized_nz_sign:1;
        uint32_t materialType:8;
      };
    };
    
    struct DeviceMaterial {
      inline DeviceMaterial() {}
      inline void operator=(const Velvet::DD &dd) { this->velvet = dd; materialType = VELVET; }
      inline __device__ bool  hasAlpha(bool isShadowRay) const;
      inline __device__ float getAlpha(vec2f tc, bool isShadowRay) const;
      inline __device__ void  make(render::HitBRDF &hit, vec3f P, vec3f N,
                                   vec2f texCoords,
                                   vec3f geometryColor, bool dbg=false) const;
      int materialType;
      union {
        AnariPhysical::DD anari;
        MiniMaterial::DD  mini;
        Velvet::DD        velvet;
      };
    };
    
  } // ::barney::render
}
