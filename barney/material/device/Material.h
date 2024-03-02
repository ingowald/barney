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
      inline __device__ EvalRes eval(render::DG dg, vec3f w_i, bool dbg=false) const;
      union {
        float3 missColor;
        render::AnariPhysical::BRDF anari;
        render::MiniMaterial::HitBSDF  mini;
        render::Velvet::HitBSDF  velvet;
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
  // ==================================================================
  // render::HitBRDF
  // ==================================================================
  
  inline __device__
  render::EvalRes render::HitBRDF::eval(render::DG dg, vec3f w_i, bool dbg) const
  {
    switch (materialType) {
    case VELVET:
      return velvet.eval(dg,w_i,dbg);
    default:
      return EvalRes(vec3f(1.f,0.f,1.f),1.f);
    }
  }
  
  inline __device__
  vec3f render::HitBRDF::getN() const
  {
    vec3f n;
    float scale = 1.f/127.f;
    n.x = quantized_nx_bits * scale;
    n.y = quantized_ny_bits * scale;
    n.z = quantized_nz_bits * scale;
    if (quantized_nx_sign) n.x = -n.x;
    if (quantized_ny_sign) n.y = -n.y;
    if (quantized_nz_sign) n.z = -n.z;
    return n;
  }

  inline __device__
  void render::HitBRDF::setDG(vec3f P, vec3f N, bool dbg)
  {
#ifdef __CUDACC__
    this->P = P;
    if (N == vec3f(0.f)) {
      quantized_nx_bits = 0;
      quantized_ny_bits = 0;
      quantized_nz_bits = 0;
      quantized_nx_sign = 0;
      quantized_ny_sign = 0;
      quantized_nz_sign = 0;
    } else {
      // N = normalize(N);
      auto quantize = [](float f) {
        return min(127,int(fabsf(f*128)));
      };
      auto sign = [](float f) {
        return f < 0.f ? 1 : 0;
      };
      quantized_nx_bits = quantize(N.x);
      quantized_ny_bits = quantize(N.y);
      quantized_nz_bits = quantize(N.z);
      quantized_nx_sign = sign(N.x);
      quantized_ny_sign = sign(N.y);
      quantized_nz_sign = sign(N.z);
    }
#endif
  }

    inline __device__
    void render::HitBRDF::setMatte(vec3f albedo, vec3f P, vec3f N)
    {
      setDG(P,N);
      materialType = render::MINI;
      mini.baseColor = albedo;
      mini.ior = 1.f;
      mini.metallic = 0.f;
      // mini.reflectance = 0.f;    
    }

    inline __device__
    vec3f render::HitBRDF::getAlbedo(bool dbg) const
    {
      switch (materialType) {
      case MINI:
        return mini.getAlbedo(dbg);
      case VELVET:
        return velvet.getAlbedo(dbg);
      default:
#ifndef NDEBUG
        printf("invalid material type in DeviceMaterial::getAlbedo...\n");
#endif
        return vec3f(0.f);
      }
      /* TODO: switch-statement over materialtype */
      return mini.baseColor;
    }


  
    // ==================================================================
    // render::DeviceMaterial (the thing that's stored in the SBT)
    // ==================================================================
  
    inline __device__
    bool render::DeviceMaterial::hasAlpha(bool isShadowRay) const
    {
      switch (materialType) {
      case MINI:
        return mini.hasAlpha(isShadowRay);
      case VELVET:
        return false;
      default:
#ifndef NDEBUG
        printf("invalid material type in DeviceMaterial::hasAlpha...\n");
#endif
        return false;
      }
    }
  
    inline __device__
    float render::DeviceMaterial::getAlpha(vec2f tc, bool isShadowRay) const
    {
      switch (materialType) {
      case MINI:
        return mini.getAlpha(tc,isShadowRay);
      case VELVET:
        return 1.f;
      default:
#ifndef NDEBUG
        printf("invalid material type in DeviceMaterial::getAlpha...\n");
#endif
        return 1.f;
      }
    }
  
    inline __device__
    void render::DeviceMaterial::make(render::HitBRDF &hit, vec3f P, vec3f N,
                                      vec2f tc,
                                      vec3f geometryColor,
                                      bool dbg) const
    {
      hit.setDG(P,N,dbg);
      hit.materialType = materialType;
      switch (materialType) {
      case MINI:
        mini.make(hit.mini,tc,geometryColor,dbg);
        break;
      case VELVET:
        if (dbg) printf("making velvet N %f %f %f\n",N.x,N.y,N.z);
        velvet.make(hit.velvet,dbg);
        break;
      default:
#ifndef NDEBUG
        printf("invalid material type in DeviceMaterial::make...\n");
#endif
      };
    }
}
