// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
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

#include "barney/DeviceGroup.h"
#include "barney/volume/TransferFunction.h"

namespace barney {

  /*! a grid of macro-cells, with each macro-cell storing both value
      range of the underlying field, and majorant after mapping
      through transfer functoin (though for efficiency reasons the
      value ranges and majorants are stored in separate arrasy). This
      class is "passive" in the sense that it only serves as a tool
      for others to create, resize, traverse, etc; it does not do any
      of that by itself (it cannot know if an accelerator wants to
      traverse it via DDA or build an optx BVH over active cells, for
      example), but provides a common interface for different fields
      to be able to compute and store macro-cells for accelerators to
      do something with. The only thing a MC grid can "acively" do is
      map its scalar ranges through a transfer functoin to compute the
      majorants. */
  struct MCGrid {
    /*! device data for this class - grid of per-cell ranges, grid of
      majorants, and dimensionality of grid */
    struct DD {
      float   *majorants;
      // not actually exported to optix programs, only used by cuda
      // kernels
      range1f *scalarRanges;
      vec3i    dims;
      vec3f    gridOrigin;
      vec3f    gridSpacing;

      inline __device__ int numCells() const
      { return dims.x*dims.y*dims.z; }
      
      inline __device__ vec3i cellID(int linearID) const
      {
        vec3i mcID;
        mcID.x = linearID % dims.x;
        mcID.y = (linearID / dims.x) % dims.y;
        mcID.z = linearID / (dims.x*dims.y);
        return mcID;
      }
      
      inline __device__
      float majorant(vec3i cellID) const
      { return majorants[cellID.x+dims.x*(cellID.y+dims.y*cellID.z)]; }
      
      /*! returns the bounding box of the given cell */
      inline __device__ box3f cellBounds(vec3i cellID,
                                         const box3f &worldBounds) const
      {
        box3f bounds;
        bounds.lower = gridOrigin + vec3f(cellID)*gridSpacing;
        bounds.upper = min(bounds.lower+gridSpacing,worldBounds.upper);
        return bounds;
      }
      
      static void addVars(std::vector<OWLVarDecl> &vars, int base);
    };
    
    MCGrid(DevGroup *devGroup);
    
    /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
    DD getDD(const std::shared_ptr<Device> &device) const;

    void setVariables(OWLGeom geom);
    
    /*! allocate memory for the given grid */
    void resize(vec3i dims);

    /*! re-set all cells' ranges to "infinite empty" */
    void clearCells();
    
    /*! given the current per-cell scalar ranges, map each such cell's
        range through the transfer functoin to compute a majorant */
    void computeMajorants(const TransferFunction *xf);

    /*! checks if this macro-cell grid has already been
        allocated/built - mostly for sanity checking nd debugging */
    inline bool built() const { return (dims != vec3i(0)); }
    
    /* buffer of range1f's, the min/max scalar values per cell */
    OWLBuffer scalarRangesBuffer = 0;
    /* buffer of floats, the actual per-cell majorants */
    OWLBuffer majorantsBuffer = 0;
    vec3i     dims { 0,0,0 };
    vec3f     gridOrigin;
    vec3f     gridSpacing;
    DevGroup *const devGroup;
  };
  
}


