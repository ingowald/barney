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

#include "barney/Object.h"
#include "barney/material/device/Material.h"

namespace barney {
  /*! barney 'virtual' material implementation that takes anari-like
      material paramters, and then builder barney::render:: style
      device materials to be put into the device geometries */
  struct Material : public SlottedObject {
    typedef std::shared_ptr<Material> SP;

    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "<Material>"; }

    /*! device-data, as a union of _all_ possible device-side
        materials; we have to use a union here because no matter what
        virtual barney::Material gets created on the host, we have to
        have a single struct we put into the OWLGeom/SBT entry, else
        we'd have to have different OWLGeom type for different
        materials .... and possibly even change the actual OWLGeom
        (and even worse, its type) if the assigned material's type
        changes */
    using DD = barney::render::DeviceMaterial;

    Material(ModelSlot *owner) : SlottedObject(owner) {}
    virtual ~Material() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    /*! @} */
    // ------------------------------------------------------------------
    static Material::SP create(ModelSlot *dg, const std::string &type);
    
    void setDeviceDataOn(OWLGeom geom) const;
    
    virtual void createDD(DD &dd, int deviceID) const = 0;

    /*! declares the device-data's variables to an owl geom */
    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    
  };

  // ==================================================================
  // render::HitBRDF
  // ==================================================================
  
  inline __device__
  vec3f render::HitBRDF::eval(render::DG dg, vec3f w_i, bool dbg) const
  {
    return mini.baseColor;
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
        return (__float_as_int(f) >> 31) & 1;
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
    /* TODO: switch-statement over materialtype */
    return mini.baseColor;
  }


  
  // ==================================================================
  // Material::DD (the thing that's stored in the SBT)
  // ==================================================================
  
  inline __device__
  bool Material::DD::hasAlpha(bool isShadowRay) const
  {
    /* TODO: switch-statement over materialtype */
    return mini.colorTexture || mini.alphaTexture
      || (isShadowRay && mini.transmission > 0.f);
  }
  
  inline __device__
  float Material::DD::getAlpha(vec2f tc, bool isShadowRay) const
  {
#ifdef __CUDACC__
    /* TODO: switch-statement over materialtype */
    if (this->mini.alphaTexture)
      return tex2D<float4>(this->mini.alphaTexture,tc.x,tc.y).w;
    if (this->mini.colorTexture)
      return tex2D<float4>(this->mini.colorTexture,tc.x,tc.y).w;
    if (isShadowRay)
      return 1.f - this->mini.transmission;
#endif
    return 1.f;
  }
  
  inline __device__
  void Material::DD::make(render::HitBRDF &hit, vec3f P, vec3f N,
                          vec2f tc,
                          vec3f geometryColor,
                          bool dbg) const
  {
    hit.setDG(P,N,dbg);
#ifdef __CUDACC__
    /* TODO: switch-statement over materialtype */
    hit.materialType = render::MINI;
    hit.mini.baseColor = mini.baseColor;
    hit.mini.ior = mini.ior;
    hit.mini.transmission = mini.transmission;
    if (!isnan(geometryColor.x)) {
      hit.mini.baseColor = geometryColor;
    } else if (this->mini.colorTexture) {
      float4 fromTex = tex2D<float4>(this->mini.colorTexture,tc.x,tc.y);
      hit.mini.baseColor = (vec3f&)fromTex;
    } else {
      hit.mini.baseColor = this->mini.baseColor;
    }
#endif
  }
}
