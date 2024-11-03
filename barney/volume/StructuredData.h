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

#include "barney/ModelSlot.h"
#include "barney/common/Texture.h"
#include "barney/volume/MCAccelerator.h"

namespace barney {

  struct ModelSlot;

  // ==================================================================
  /*! Scalar field made of 3D structured data, constting of Nx*Ny*Nz
      scalars.

      Supported settable fields:

      - "texture" (BNTexture) : a Texture3D with the actual scalars
         
      - "dims" (int3) : 3D array dimensions of the scalars

      - "gridOrigin" (float3) : world-space origin of the scalar
      grid positions

      - "gridSpacing" (float3) : world-space spacing of the scalar
      grid positions

      - (experimental) "colorMapTexture" : _additional_ 3D RGB texture
        that "overwrites" the RGB component of the transfer function -
        ie, for the final RGBA sample 'a' comes from the transfer
        functoin, while RGB comes from that 3D texture
  */
  struct StructuredData : public ScalarField
  {
    /*! device data for this class; note we specify both the 'usual'
        texture object (which does trilinerar interpolation) as well
        as a "NN" (nearest) sampling for the macro-cell generation */
    struct DD : public ScalarField::DD {
      cudaTextureObject_t texObj;
      vec3f cellGridOrigin;
      vec3f cellGridSpacing;
      vec3i numCells;
      cudaTextureObject_t colorMappingTexObj;

#ifdef __CUDA_ARCH__
      /*! "template-virtual" function that a sampler calls on an
          _already_ transfer-function mapped RGBA value, allowing the
          scalar field to do some additional color mapping on top of
          whatever came out of the transfer function. the default
          implementation (provided here int his base class coommon to
          all scalar fields) is to just return the xf-color mapped
          RBGA value */
      inline __device__ vec4f mapColor(vec4f xfColorMapped,
                                       vec3f P, float scalar) const
      {
        if (!colorMappingTexObj) return xfColorMapped;
        vec3f rel = (P - cellGridOrigin) * rcp(cellGridSpacing);
        float4 fromColorMap = tex3D<float4>(colorMappingTexObj,rel.x,rel.y,rel.z);
        fromColorMap.w = xfColorMapped.w;
        return fromColorMap;
      }
#endif

      
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
    };

    /*! construct a new structured data scalar field; will not do
        anything - or have any data - untile 'set' and 'commit'ed
    */
    StructuredData(Context *context, int slot);
    virtual ~StructuredData() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set3i(const std::string &member, const vec3i &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    void commit() override;
    /*! @} */
    // ------------------------------------------------------------------
    
    void setVariables(OWLGeom geom);
    VolumeAccel::SP createAccel(Volume *volume) override;
    void buildMCs(MCGrid &macroCells) override;

    Texture3D::SP  texture;
    Texture3D::SP  colorMapTexture;
    BNDataType   scalarType = BN_DATA_UNDEFINED;
    vec3i numScalars  { 0,0,0 };
    vec3i numCells    { 0,0,0 }; 
    vec3f gridOrigin  { 0,0,0 };
    vec3f gridSpacing { 1,1,1 };
  };

  /*! for structured data, the sampler doesn't have to do much but
      sample the 3D texture that the structeuddata field has already
      created. in thoery one could argue that the 3d texture should
      belong ot the sampler (not the field), but the field needs it to
      compute the macro cells, so we'll leave it as such for now */
  struct StructuredDataSampler {
    struct DD : public StructuredData::DD {
      inline __device__ float sample(const vec3f P, bool dbg) const;
      // inline __device__ vec4f mapColor(vec4f xfColorMapped,
      //                                  vec3f point, float scalar) const
      // { return xfColorMapped; }
    };

    struct Host
    {
      Host(ScalarField *field)
        : field((StructuredData *)field)
      {}

      /*! builds the string that allows for properly matching optix
        device progs for this type */
      inline std::string getTypeString() const { return "Structured"; }

      /*! doesn'ta ctualy do anything for this class, but required to
          make the template instantiating it happy */
      void setVariables(OWLGeom geom) { /* nothing to do for this class */}
      
      /*! doesn'ta ctualy do anything for this class, but required to
          make the template instantiating it happy */
      void build(bool full_rebuild) { /* nothing to do for this class */}
      
      StructuredData *const field;
    };
  };
  
#ifdef __CUDA_ARCH__
  inline __device__ float StructuredDataSampler::DD::sample(const vec3f P, bool dbg) const
  {
    vec3f rel = (P - cellGridOrigin) * rcp(cellGridSpacing);
        
    if (rel.x < 0.f) return NAN;
    if (rel.y < 0.f) return NAN;
    if (rel.z < 0.f) return NAN;
    if (rel.x >= numCells.x) return NAN;
    if (rel.y >= numCells.y) return NAN;
    if (rel.z >= numCells.z) return NAN;
    float f = tex3D<float>(texObj,rel.x+.5f,rel.y+.5f,rel.z+.5f);
    return f;
  }
#endif
}


