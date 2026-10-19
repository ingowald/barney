// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_rtc.h"
#if BARNEY_HAVE_NANOVDB
#include "native/ModelSlot.h"
#include "native/common/Texture.h"
#include "native/volume/MCAccelerator.h"
// nanovdb
#include <nanovdb/NanoVDB.h>
#include "nanovdb/math/Math.h"
#if !defined(BARNEY_DEVICE_PROGRAM)
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#endif

// Each entry is MACRO(BuildType, Suffix, EnumName) where:
//   BuildType — the nanovdb build type (template argument to NanoTree<>)
//   Suffix    — a token used in symbol names (e.g., device program names)
//   EnumName  — the nanovdb::GridType enumerator name
#define BARNEY_NANOVDB_FLOAT_TYPES(MACRO) \
  MACRO(float, float, Float)

namespace BARNEY_NS {
  namespace native {

    struct NanoVDBData : public ScalarField
    {
      /*! device data for this class. nothis special for a structured
        data object; all sampling related stuff will be in the
        StructuredDataSampler */
      struct DD : public ScalarField::DD {
        vec3f voxelSize;
        vec3i gridSize;
        box3i indexBounds;
        nanovdb::GridType gridType;
        const void *gridData;
      };

      /*! construct a new NanoDVB scalar field; will not do anything -
        or have any data - untile 'set' and 'commit'ed
      */
      NanoVDBData(Context *context,
                  const DevGroup::SP &devices);
      virtual ~NanoVDBData() = default;

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      bool setData(const std::string &member, const Data::SP &value) override;
      void commit() override;
      /*! @} */
      // ------------------------------------------------------------------
    
      DD getDD(Device *device);
      VolumeAccel::SP createAccel(Volume *volume) override;
      IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;
      MCGrid::SP buildMCs() override;

      PODData::SP data;
      box3i indexBounds;
      vec3f voxelSize;
      vec3i gridSize;
      range1f valueRange;
      nanovdb::GridType gridType;
    };

    /*! sampler object for a StructuredData object, using a 3d texture */
    template<typename T>
    struct NanoVDBDataSampler : public ScalarFieldSampler {
      using NVDBGridT = typename nanovdb::Grid<nanovdb::NanoTree<T>>;
      using NVDBAccessorT = typename NVDBGridT::AccessorType;
    
      NanoVDBDataSampler(NanoVDBData *const sf)
        : sf(sf)
      {}
    
      struct DD {
#if RTC_DEVICE_CODE
        inline __rtc_device float sample(const vec3f P, bool dbg=false) const;
#endif
        NVDBGridT *nvdbGrid;
        box3i indexBounds;
      };

      void build() override;
    
      DD getDD(Device *device);
      NanoVDBData *const sf;
    };
  
#if RTC_DEVICE_CODE
    template<typename T>
    inline __rtc_device
    float NanoVDBDataSampler<T>::DD::sample(const vec3f P, bool dbg) const
    {
      const auto nvdbLoc = nanovdb::Vec3f(P.x, P.y, P.z);

      // NanoVDB's worldBBox max is computed as map(indexBBox.max + 1), i.e. the
      // upper corner of the last voxel, not its center. So worldToIndexF maps
      // a world position to the voxel's lower-left corner index. We subtract 0.5
      // to shift into voxel-center index coordinates for correct trilinear
      // sampling. This matches the workaround in VisRTX for cell-centered NanoVDB.
      auto indexPos = nvdbGrid->worldToIndexF(nvdbLoc);
      indexPos[0] -= 0.5;
      indexPos[1] -= 0.5;
      indexPos[2] -= 0.5;

      // Clamp to the valid voxel range so boundary rays extrapolate the outermost
      // voxel value rather than blending with the background (inactive) value.
      indexPos[0] = nanovdb::math::Clamp(indexPos[0],
                                         (float)indexBounds.lower.x, (float)indexBounds.upper.x);
      indexPos[1] = nanovdb::math::Clamp(indexPos[1],
                                         (float)indexBounds.lower.y, (float)indexBounds.upper.y);
      indexPos[2] = nanovdb::math::Clamp(indexPos[2],
                                         (float)indexBounds.lower.z, (float)indexBounds.upper.z);

      int x0 = nanovdb::math::Floor(indexPos[0]);
      int y0 = nanovdb::math::Floor(indexPos[1]);
      int z0 = nanovdb::math::Floor(indexPos[2]);
      const float fx = float(indexPos[0] - x0);
      const float fy = float(indexPos[1] - y0);
      const float fz = float(indexPos[2] - z0);
      const int x1 = x0 < indexBounds.upper.x ? x0 + 1 : x0;
      const int y1 = y0 < indexBounds.upper.y ? y0 + 1 : y0;
      const int z1 = z0 < indexBounds.upper.z ? z0 + 1 : z0;

      auto acc = nvdbGrid->getAccessor();
      const float v000 = acc.getValue(nanovdb::Coord(x0, y0, z0));
      const float v100 = acc.getValue(nanovdb::Coord(x1, y0, z0));
      const float v010 = acc.getValue(nanovdb::Coord(x0, y1, z0));
      const float v110 = acc.getValue(nanovdb::Coord(x1, y1, z0));
      const float v001 = acc.getValue(nanovdb::Coord(x0, y0, z1));
      const float v101 = acc.getValue(nanovdb::Coord(x1, y0, z1));
      const float v011 = acc.getValue(nanovdb::Coord(x0, y1, z1));
      const float v111 = acc.getValue(nanovdb::Coord(x1, y1, z1));

      const float v00 = v000 + fx * (v100 - v000);
      const float v10 = v010 + fx * (v110 - v010);
      const float v01 = v001 + fx * (v101 - v001);
      const float v11 = v011 + fx * (v111 - v011);
      const float v0 = v00 + fy * (v10 - v00);
      const float v1 = v01 + fy * (v11 - v01);
      return v0 + fz * (v1 - v0);
    }
#endif
  }
}

#endif

