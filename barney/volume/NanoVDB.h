// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/ModelSlot.h"
#include "barney/common/Texture.h"
#include "barney/volume/MCAccelerator.h"
// nanovdb
#include <nanovdb/NanoVDB.h>
#include "nanovdb/math/Math.h"
#include "nanovdb/math/SampleFromVoxels.h"
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Math.h>

// glm
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/component_wise.hpp>


namespace BARNEY_NS {

  struct ModelSlot;

  struct NanoVDBData : public ScalarField
  {
    /*! device data for this class. nothis special for a structured
        data object; all sampling related stuff will be in the
        StructuredDataSampler */
    struct DD : public ScalarField::DD {
      // box3f worldBounds;
      vec3f voxelSize;
      vec3i gridSize;
      nanovdb::GridType gridType;
      const void *gridData;
      /* nothing */
    };

    // struct PLD {
    //   // DeviceBuffer m_deviceBuffer;
    //   void *gridData;
    // };
    // PLD *getPLD(Device *device);
    // std::vector<PLD> perLogical;
    
    /*! construct a new structured data scalar field; will not do
        anything - or have any data - untile 'set' and 'commit'ed
    */
    NanoVDBData(Context *context,
                       const DevGroup::SP &devices);
    virtual ~NanoVDBData() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1i(const std::string &member, const int &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    bool setData(const std::string &member, const Data::SP &value) override;
    void commit() override;
    /*! @} */
    // ------------------------------------------------------------------
    
    DD getDD(Device *device);
    VolumeAccel::SP createAccel(Volume *volume) override;
    IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;
    MCGrid::SP buildMCs() override;

    // TextureData::SP scalars;
    // Texture::SP  texture;
    // Texture::SP  textureNN;

    // BNDataType scalarType = BN_DATA_UNDEFINED;
    // vec3i numScalars  { 0,0,0 };
    // vec3i numCells    { 0,0,0 }; 
    // vec3f gridOrigin  { 0,0,0 };
    // vec3f gridSpacing { 1,1,1 };
    PODData::SP data;
    // box3f worldBounds;
    vec3f voxelSize;
    vec3i gridSize;
    nanovdb::GridType gridType;
  };

  /*! sampler object for a StructuredData object, using a 3d texture */
  template<typename T>
  struct NanoVDBDataSampler : public ScalarFieldSampler {
    using NVDBGridT = typename nanovdb::Grid<nanovdb::NanoTree<T>>;
    using NVDBAccessorT = typename NVDBGridT::AccessorType;
    using NVDBSamplerT = nanovdb::math::SampleFromVoxels<NVDBAccessorT, 1>;
    
    NanoVDBDataSampler(NanoVDBData *const sf)
      : sf(sf)
    {}
    
    struct DD {
      NVDBGridT *nvdbGrid;
      // AccessorType m_accessor;
      // SamplerType m_sampler;
      
#if RTC_DEVICE_CODE
      inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
#endif
      
      // box3f worldBounds;
      vec3f voxelSize;
      // const void *gridData;

      // : m_grid(
      //       reinterpret_cast<const GridType *>(sf.data.nvdbRegular.gridData)),
      //   m_accessor(m_grid->getAccessor()),
      //   m_sampler(nanovdb::math::createSampler<1>(m_accessor))
      
      // nanovdb::GridType gridType;
      // AccessorType m_accessor;
      // SamplerType m_sampler;
      // rtc::TextureObject texObj;
      // vec3f cellGridOrigin;
      // vec3f cellGridSpacing;
      // vec3i numCells;
    };

    void build() override;
    
    DD getDD(Device *device);
    NanoVDBData *const sf;// sf.data.nvdbRegular
  };
  
#if RTC_DEVICE_CODE
  template<typename T>
  inline __rtc_device
  float NanoVDBDataSampler<T>::DD::sample(const vec3f P, bool dbg) const
  {
#if 1
    const auto nvdbLoc = nanovdb::Vec3d(P.x, P.y, P.z);
    
    auto acc = nvdbGrid->getAccessor();
    // auto sampler = nanovdb::math::createSampler<0>(acc);
    auto sampler = nanovdb::math::createSampler<1>(acc);
    float res = sampler(nanovdb::math::Vec3d(nvdbGrid->worldToIndexF(nvdbLoc)));
    if (dbg)
      printf("sample %f %f %f -> %f\n",
             P.x,P.y,P.z,res);
    return res;
    // printf("TODO nanovdb\n");
    // auto acc = m_grid->getAccessor();
#else
    auto acc = sf.asNanoVDB.grid->getAccessor();
    if (sf.asNanoVDB.filterMode == Nearest) {
      auto smp = nanovdb::math::createSampler<0>(acc);
      value = smp(nanovdb::math::Vec3<float>(P.x,P.y,P.z));
      primID = 0;
      return true;
    } else if (sf.asNanoVDB.filterMode == Linear) {
      auto smp = nanovdb::math::createSampler<1>(acc);
      value = smp(nanovdb::math::Vec3<float>(P.x,P.y,P.z));
      primID = 0;
      return true;
    }
#endif
    // vec3f rel = (P - cellGridOrigin) * rcp(cellGridSpacing);
    // if (rel.x < 0.f) return NAN;
    // if (rel.y < 0.f) return NAN;
    // if (rel.z < 0.f) return NAN;
    // if (rel.x >= numCells.x) return NAN;
    // if (rel.y >= numCells.y) return NAN;
    // if (rel.z >= numCells.z) return NAN;
    // float f = rtc::tex3D<float>(texObj,rel.x+.5f,rel.y+.5f,rel.z+.5f);
    // return f;
  }
#endif
}


