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

namespace BARNEY_NS {

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
  */
  struct StructuredData : public ScalarField
  {
    /*! device data for this class. nothis special for a structured
        data object; all sampling related stuff will be in the
        StructuredDataSampler */
    struct DD : public ScalarField::DD {
      /* nothing */
    };

    /*! construct a new structured data scalar field; will not do
        anything - or have any data - untile 'set' and 'commit'ed
    */
    StructuredData(Context *context,
                   const DevGroup::SP &devices);
    virtual ~StructuredData() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set3i(const std::string &member, const vec3i &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    void commit() override;
    /*! @} */
    // ------------------------------------------------------------------
    
    DD getDD(Device *device);
    VolumeAccel::SP createAccel(Volume *volume) override;
    void buildMCs(MCGrid &macroCells) override;

    Texture3D::SP  texture;
    // Texture3D::SP  colorMapTexture;

    struct PLD {
      rtc::ComputeKernel3D *computeMCs = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank]; } 
    std::vector<PLD> perLogical;
    
    BNDataType scalarType = BN_DATA_UNDEFINED;
    vec3i numScalars  { 0,0,0 };
    vec3i numCells    { 0,0,0 }; 
    vec3f gridOrigin  { 0,0,0 };
    vec3f gridSpacing { 1,1,1 };
  };

  /*! sampler object for a StructreData object, using a 3d texture */
  struct StructuredDataSampler : public ScalarFieldSampler {
    StructuredDataSampler(StructuredData *const sf)
      : sf(sf)
    {}
    
    struct DD {
      inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
      
      rtc::device::TextureObject texObj;
      vec3f cellGridOrigin;
      vec3f cellGridSpacing;
      vec3i numCells;
    };

    void build() override {}
    
    DD getDD(Device *device);
    StructuredData *const sf;
  };
#if RTC_DEVICE_CODE
  inline __rtc_device
  float StructuredDataSampler::DD::sample(const vec3f P, bool dbg) const
  {
    vec3f rel = (P - cellGridOrigin) * rcp(cellGridSpacing);
    if (rel.x < 0.f) return NAN;
    if (rel.y < 0.f) return NAN;
    if (rel.z < 0.f) return NAN;
    if (rel.x >= numCells.x) return NAN;
    if (rel.y >= numCells.y) return NAN;
    if (rel.z >= numCells.z) return NAN;
    float f = rtc::tex3D<float>(texObj,rel.x+.5f,rel.y+.5f,rel.z+.5f);
    return f;
  }
#endif
}


