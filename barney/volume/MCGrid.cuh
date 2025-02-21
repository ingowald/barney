// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

/*! \file MCGrid.cuh Helper functions for building macro-cell grids,
  basically to allow for atomic min/max updates when 'rasterizing'
  primitives into a grid */
#pragma once

#include "barney/volume/MCGrid.h"
#include "barney/common/barney-common.h"

namespace BARNEY_NS {

#if RTC_DEVICE_CODE
  // -----------------------------------------------------------------------------
  // INTERFACE
  // -----------------------------------------------------------------------------

  inline __device__
  int project(float f,
              const range1f range,
              int dim);

  /*! projects a given position into a grid defined by world-space
      'bounds' and dimensions 'dims', and return the cell that this
      world-sapce point projects to */
  inline __device__
  vec3i project(const vec3f &pos,
                const box3f &bounds,
                const vec3i &dims);

  /*! rasters a given 4D-(space-and-value)-primitive into the given
      grid; computing all grid cells that this prim covers, and doing,
      for each cell, an atomin min/max based on the prim's value range
      (its min/max .w values) */
  inline __device__
  void rasterBox(MCGrid::DD grid,
                 const box3f worldBounds,
                 const box4f primBounds4);

  // -----------------------------------------------------------------------------
  // IMPLEMENTATION
  // -----------------------------------------------------------------------------

  inline __device__
  int project(float f,
              const range1f range,
              int dim)
  {
    return max(0,min(dim-1,int(dim*(f-range.lower)/(range.upper-range.lower))));
  }

  inline __device__
  vec3i project(const vec3f &pos,
                const box3f &bounds,
                const vec3i &dims)
  {
    return vec3i(project(pos.x,{bounds.lower.x,bounds.upper.x},dims.x),
                 project(pos.y,{bounds.lower.y,bounds.upper.y},dims.y),
                 project(pos.z,{bounds.lower.z,bounds.upper.z},dims.z));
  }

  inline __device__
  void rasterBox(MCGrid::DD grid,
                 const box3f worldBounds,
                 const box4f primBounds4)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = vec3i((pb.lower-grid.gridOrigin)*rcp(grid.gridSpacing));
    vec3i hi = vec3i((pb.upper-grid.gridOrigin)*rcp(grid.gridSpacing));

    lo = min(max(lo,vec3i(0)),grid.dims-vec3i(1));
    hi = min(max(hi,vec3i(0)),grid.dims-vec3i(1));

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const int cellID
            = ix
            + iy * grid.dims.x
            + iz * grid.dims.x * grid.dims.y;
          auto &cell = grid.scalarRanges[cellID];
          rtc::fatomicMin(&cell.lower,primBounds4.lower.w);
          rtc::fatomicMax(&cell.upper,primBounds4.upper.w);
        }
  }

  inline __device__
  void rasterBox(MCGrid::DD grid,
                 const box4f primBounds4)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = vec3i((pb.lower-grid.gridOrigin)*rcp(grid.gridSpacing));
    vec3i hi = vec3i((pb.upper-grid.gridOrigin)*rcp(grid.gridSpacing));

    lo = min(max(lo,vec3i(0)),grid.dims-vec3i(1));
    hi = min(max(hi,vec3i(0)),grid.dims-vec3i(1));

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const int cellID
            = ix
            + iy * grid.dims.x
            + iz * grid.dims.x * grid.dims.y;
          auto &cell = grid.scalarRanges[cellID];
          rtc::fatomicMin(&cell.lower,primBounds4.lower.w);
          rtc::fatomicMax(&cell.upper,primBounds4.upper.w);
        }
  }
#endif
  
}
