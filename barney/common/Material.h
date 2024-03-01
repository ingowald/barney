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

namespace barney {
  namespace render {
    
    struct DG {
      vec3f N;
      vec3f w_o;
    };

    /*! device-side implementation of anari "physical" material */
    struct AnariPhysical {
      struct BRDF {
        /*! "BRDFs" are the thing that goes onto a ray, and is used for
          sampling, eval, etc */
        inline __device__ vec3f eval(DG dg, vec3f L_i, vec3f w_i) const;
        inline __device__ vec3f getAlbedo() const;
        vec3h baseColor;
        half  ior;
        half  metallic;
        vec3h reflectance;
      };
      /*! "DDs" are the device-data that gets stored in the associated
        geometry's SBT entry */
      struct DD {
        vec3f baseColor;
        float ior;
        float transmission;
        float roughness;
        float metallic;
        cudaTextureObject_t colorTexture;
        cudaTextureObject_t alphaTexture;
      };
    }; // ::barney::render::AnariPhysical

  } // ::barney::render

  typedef enum { INVALID=0, PHYSICAL } MaterialType;
  
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
      inline __device__ void modulateBaseColor(vec3f rbga);
      union {
        render::AnariPhysical::BRDF anari;
      };
      vec3f P;
      uint8_t quantizedNormal[3];
      uint8_t type;
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
      inline __device__ bool  hasAlpha() const;
      inline __device__ float getAlpha(vec2f tc) const;
      inline __device__ vec3f getAlbedo() const;
     inline __device__ void make(HitBRDF &hit, vec3f P, vec3f N,
                                  vec2f texCoords,
                                  vec3f colorFromTexture) const;
      int type;
      union {
        render::AnariPhysical::DD anari;
      };
    };

    Material(ModelSlot *owner) : SlottedObject(owner) {}
    virtual ~Material() = default;

    static Material::SP create(ModelSlot *dg, const std::string &type);
    
    void setDeviceDataOn(OWLGeom geom) const;
    
    virtual void createDD(DD &dd, int deviceID) = 0;

    /*! declares the device-data's variables to an owl geom */
    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    
  };

  struct AnariPhysicalMaterial : public barney::Material {
    AnariPhysicalMaterial(ModelSlot *owner) : Material(owner) {}
    virtual ~AnariPhysicalMaterial() = default;
    
    void createDD(DD &dd, int deviceID) override;
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
    void createDD(DD &dd, int deviceID) override;
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
  
}
