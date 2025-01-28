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

#include "barney/volume/MCGrid.h"

namespace barney {

  MCGrid::MCGrid(const DevGroup::SP &devices)
    : devices(devices)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      
      pld->scalarRangesBuffer = rtc->createBuffer(sizeof(float2));
      pld->majorantsBuffer    = rtc->createBuffer(sizeof(float));

      pld->mapMCs             = rtc->createCompute("mapMCs");
      pld->clearMCs           = rtc->createCompute("clearMCs");
    }
  }
                            

  struct ClearMCs {
    /* kernel ARGS */
    MCGrid::DD grid;

    /* kernel CODE */
    template<typename RTCore>
    inline __both__
    void run(const RTCore &rtCore)
    {
      int ix = rtCore.getThreadIdx().x
        +rtCore.getBlockIdx().x*rtCore.getBlockDim().x;
      if (ix >= grid.dims.x) return;
      
      int iy = rtCore.getThreadIdx().y
        +rtCore.getBlockIdx().y*rtCore.getBlockDim().y;
      if (iy >= grid.dims.y) return;
      
      int iz = rtCore.getThreadIdx().z
        +rtCore.getBlockIdx().z*rtCore.getBlockDim().z;
      if (iz >= grid.dims.z) return;
      
      
      int ii = ix + grid.dims.x*(iy + grid.dims.y*(iz));
      grid.scalarRanges[ii] = { +BARNEY_INF, -BARNEY_INF };
    }
  };
  
  /*! re-set all cells' ranges to "infinite empty" */
  void MCGrid::clearCells()
  {
    const vec3i bs = 4;
    const vec3i nb = divRoundUp(dims,bs);
    for (auto device : *devices) {
      auto d_grid = getDD(device);
      ClearMCs args = { d_grid };
      getPLD(device)->clearMCs->launch(nb,bs,&args);
    }
    for (auto device : *devices) 
      device->sync();
  }
  
  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  struct MapMCs {
    /* kernel ARGS */
    MCGrid::DD grid;
    TransferFunction::DD xf;
    
    /* kernel CODE */
    template<typename RTCore>
    inline __both__
    void run(const RTCore &rtCore)
    {
      int ix = rtCore.getThreadIdx().x
        +rtCore.getBlockIdx().x*rtCore.getBlockDim().x;
      if (ix >= grid.dims.x) return;
      
      int iy = rtCore.getThreadIdx().y
        +rtCore.getBlockIdx().y*rtCore.getBlockDim().y;
      if (iy >= grid.dims.y) return;
      
      int iz = rtCore.getThreadIdx().z
        +rtCore.getBlockIdx().z*rtCore.getBlockDim().z;
      if (iz >= grid.dims.z) return;
      
      vec3i mcID(ix,iy,iz);
      
      int mcIdx = mcID.x + grid.dims.x*(mcID.y + grid.dims.y*mcID.z);
      range1f scalarRange = grid.scalarRanges[mcIdx];
      const float maj = xf.majorant(scalarRange);
      // printf("mapmc %f %f in %f %f -> %f\n",
      //        scalarRange.lower,
      //        scalarRange.upper,
      //        xf.domain.lower,
      //        xf.domain.upper,
      //        maj);
      grid.majorants[mcIdx] = maj;
    }
  };
  
  /*! recompute all macro cells' majorant value by remap each such
    cell's value range through the given transfer function */
  void MCGrid::computeMajorants(TransferFunction *xf)
  {
    assert(xf);
    assert(dims.x > 0);
    assert(dims.y > 0);
    assert(dims.z > 0);
    const vec3i bs = 4;
    // cuda num blocks
    const vec3i nb = divRoundUp(dims,bs);

    for (auto device : *devices) {
      auto d_xf = xf->getDD(device);
      auto dd = getDD(device);
      MapMCs args = { dd,d_xf };
      getPLD(device)->mapMCs->launch(nb,bs,&args);
    }
    for (auto device : *devices) 
      device->sync();
  }
  
  /*! allocate memory for the given grid */
  void MCGrid::resize(vec3i dims)
  {
    assert(dims.x > 0);
    assert(dims.y > 0);
    assert(dims.z > 0);
    this->dims = dims;
    size_t numCells = owl::common::volume(dims);
    
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      rtc->freeBuffer(pld->majorantsBuffer);
      rtc->freeBuffer(pld->scalarRangesBuffer);
      
      pld->majorantsBuffer
        = rtc->createBuffer(sizeof(float)*numCells);
      pld->scalarRangesBuffer
        = rtc->createBuffer(sizeof(float2)*numCells);
    }
    for (auto device : *devices) 
      device->sync();
  }
  
  /*! get cuda-usable device-data for given device ID (relative to
    devices in the devgroup that this gris is in */
  MCGrid::DD MCGrid::getDD(Device *device) 
  {
    PLD *pld = getPLD(device);
    
    MCGrid::DD dd;
    
    dd.majorants
      = (float *)pld->majorantsBuffer->getDD();
    dd.scalarRanges
      = (range1f*)pld->scalarRangesBuffer->getDD();

    dd.dims = dims;
    dd.gridOrigin = gridOrigin;
    dd.gridSpacing = gridSpacing;
    return dd;
  }
  
} // ::barney



RTC_DECLARE_COMPUTE(clearMCs,barney::ClearMCs);
RTC_DECLARE_COMPUTE(mapMCs,barney::MapMCs);
