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

#include "barney/common/Texture.h"
#include "barney/common/Data.h"
#include "barney/common/half.h"

namespace barney {
  namespace render {
    
    typedef enum { MISS=0, MINI, ANARI_PHYSICAL } MaterialType;
  
    struct DG {
      vec3f N;
      vec3f w_o;
    };

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
    
    struct MiniMaterial {
      struct BRDF {
        inline __device__ vec3f eval(DG dg, vec3f w_i) const;
        vec3h baseColor;
        half  ior;
        half  metallic;
        half  transmission;
      };
      struct DD {
        vec3f baseColor;
        float ior;
        float transmission;
        // float roughness;
        float metallic;
        cudaTextureObject_t colorTexture;
        cudaTextureObject_t alphaTexture;
      };
    };
  } // ::barney::render

  /*! barney 'virtual' material implementation that takes anari-like
      material paramters, and then builder barney::render:: style
      device materials to be put into the device geometries */
  struct Material : public SlottedObject {
    typedef std::shared_ptr<Material> SP;

    struct HitBRDF {
      /*! helper function to set this to a matte material, primarily
          for volume data */
      inline __device__ void setMatte(vec3f albedo, vec3f P, vec3f N);
      /*! modulate given BRDF with a color form texture, or colors[] array, etc */
      // inline __device__ void modulateBaseColor(vec3f rbga);
      inline __device__ void setDG(vec3f P, vec3f N);
      inline __device__ vec3f getAlbedo() const;
      inline __device__ vec3f getN() const;
      inline __device__ vec3f eval(render::DG dg, vec3f w_i) const;
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
    
    /*! device-data, as a union of _all_ possible device-side
        materials; we have to use a union here because no matter what
        virtual barney::Material gets created on the host, we have to
        have a single struct we put into the OWLGeom/SBT entry, else
        we'd have to have different OWLGeom type for different
        materials .... and possibly even change the actual OWLGeom
        (and even worse, its type) if the assigned material's type
        changes */
    struct DD {
      inline DD() {}
      inline __device__ bool  hasAlpha(bool isShadowRay) const;
      inline __device__ float getAlpha(vec2f tc, bool isShadowRay) const;
      inline __device__ void  make(HitBRDF &hit, vec3f P, vec3f N,
                                   vec2f texCoords,
                                   vec3f geometryColor) const;
      int type;
      union {
        render::AnariPhysical::DD anari;
        render::MiniMaterial::DD  mini;
      };
    };

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

  struct AnariPhysicalMaterial : public barney::Material {
    AnariPhysicalMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~AnariPhysicalMaterial() = default;
    
    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    /* iw - i have NO CLUE what goes in here .... */
    vec3f baseColor { .5f, .5f, .5f };
  };

  /*! material according to "miniScene" default specification. will
      internally build a AnariPhyisical device data */
  struct MiniMaterial : public barney::Material {
    MiniMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~MiniMaterial() = default;
    void createDD(DD &dd, int deviceID) const override;
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    vec3f baseColor { .5f, .5f, .5f };
    float transmission { 0.f };
    float roughness    { 0.f };
    float metallic     { 0.f };
    float ior          { 1.f };
    Texture::SP colorTexture;
    Texture::SP alphaTexture;
  };

  
  // ==================================================================
  // Material::HitBRDF
  // ==================================================================
  
  inline __device__
  vec3f Material::HitBRDF::eval(render::DG dg, vec3f w_i) const
  {
    return mini.baseColor;
  }
  
  inline __device__
  vec3f Material::HitBRDF::getN() const
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
  void Material::HitBRDF::setDG(vec3f P, vec3f N)
  {
#ifdef __CUDACC__
    P = P;
    if (N == vec3f(0.f)) {
      quantized_nx_bits = 0;
      quantized_ny_bits = 0;
      quantized_nz_bits = 0;
      quantized_nx_sign = 0;
      quantized_ny_sign = 0;
      quantized_nz_sign = 0;
    } else {
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
  void Material::HitBRDF::setMatte(vec3f albedo, vec3f P, vec3f N)
  {
    setDG(P,N);
    materialType = render::MINI;
    mini.baseColor = albedo;
    mini.ior = 1.f;
    mini.metallic = 0.f;
    // mini.reflectance = 0.f;    
  }

  inline __device__
  vec3f Material::HitBRDF::getAlbedo() const
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
  void Material::DD::make(HitBRDF &hit, vec3f P, vec3f N,
                          vec2f tc,
                          vec3f geometryColor) const
  {
#ifdef __CUDACC__
    /* TODO: switch-statement over materialtype */
    hit.materialType = render::MINI;
    hit.mini.baseColor = mini.baseColor;
    hit.mini.ior = mini.ior;
    hit.mini.transmission = mini.transmission;
    if (!isnan(geometryColor.x)) {
      hit.mini.baseColor = geometryColor;
    } else if (this->mini.colorTexture) {
      tex2D<float4>(this->mini.colorTexture,tc.x,tc.y);
    } else {
      hit.mini.baseColor = this->mini.baseColor;
    }
#endif
  }
}
