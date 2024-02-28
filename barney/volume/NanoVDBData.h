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

#include "barney/DataGroup.h"
#include "barney/Texture.h"
#include "barney/volume/MCAccelerator.h"

#define NANOVDB_USE_INTRINSICS
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

namespace barney {

  struct DataGroup;

  // ==================================================================
  /*! NanoVDB data.

      Supported settable fields:

      - "texture" (BNTexture) : NanoVDB with the actual scalars
         
      - "dims" (int3) : 3D array dimensions of the scalars

      - "gridOrigin" (float3) : world-space origin of the scalar
      grid positions

      - "gridSpacing" (float3) : world-space spacing of the scalar
      grid positions
  */
  struct NanoVDBData : public ScalarField
  {
    enum InterpolationType {
      INTERPOLATION_NONE      = -1,
      INTERPOLATION_CLOSEST   = 0,
      INTERPOLATION_LINEAR    = 1,
      INTERPOLATION_QUADRATIC = 2,
      INTERPOLATION_CUBIC     = 3
    };

    /*! device data for this class */
    struct DD : public ScalarField::DD {
      void* nanogrid;
      vec3f cellGridOrigin;
      vec3f cellGridSpacing;
      vec3i numCells;
      cudaTextureObject_t colorMappingTexObj;
      int interpolation;
      
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
    };

    /*! construct a new NanoVDB data scalar field; will not do
        anything - or have any data - untile 'set' and 'commit'ed
    */
    NanoVDBData(DataGroup *owner);
    virtual ~NanoVDBData() = default;

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

    TextureNanoVDB::SP  texture;
    BNScalarType   scalarType = BN_SCALAR_UNDEFINED;
    vec3i numScalars  { 0,0,0 };
    vec3i numCells    { 0,0,0 }; 
    vec3f gridOrigin  { 0,0,0 };
    vec3f gridSpacing { 1,1,1 };
  };

  /*! for NanoVDB data, the sampler doesn't have to do much but
      sample the 3D texture that the structeuddata field has already
      created. in thoery one could argue that the 3d texture should
      belong ot the sampler (not the field), but the field needs it to
      compute the macro cells, so we'll leave it as such for now */
  struct NanoVDBDataSampler {
    struct DD : public NanoVDBData::DD {
      inline __device__ float sample(const vec3f P, bool dbg) const;
    };

    struct Host
    {
      Host(ScalarField *field)
        : field((NanoVDBData *)field)
      {}

      /*! builds the string that allows for properly matching optix
        device progs for this type */
      inline std::string getTypeString() const { return "NanoVDB"; }

      /*! doesn'ta ctualy do anything for this class, but required to
          make the template instantiating it happy */
      void setVariables(OWLGeom geom) { /* nothing to do for this class */}
      
      /*! doesn'ta ctualy do anything for this class, but required to
          make the template instantiating it happy */
      void build(bool full_rebuild) { /* nothing to do for this class */}
      
      NanoVDBData *const field;
    };
  };
  
#ifdef __CUDA_ARCH__
  inline __device__ float NanoVDBDataSampler::DD::sample(const vec3f P, bool dbg) const
  {
    vec3f rel = (P - cellGridOrigin) * rcp(cellGridSpacing);
        
    if (rel.x < 0.f) return NAN;
    if (rel.y < 0.f) return NAN;
    if (rel.z < 0.f) return NAN;
    if (rel.x >= numCells.x) return NAN;
    if (rel.y >= numCells.y) return NAN;
    if (rel.z >= numCells.z) return NAN;

    nanovdb::NanoGrid<float>* const _grid = (nanovdb::NanoGrid<float> *)nanogrid;
    typedef typename nanovdb::NanoGrid<float>::AccessorType AccessorType;
    AccessorType acc = _grid->getAccessor();

    float f = nanovdb::SampleFromVoxels<AccessorType, 1, false>(acc)(nanovdb::Vec3f(rel.x, rel.y, rel.z));

    return f;
  }
#endif
}


