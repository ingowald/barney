// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/volume/MCGrid.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  MajorantsGrid::MajorantsGrid(MCGrid::SP mcGrid)
    : mcGrid(mcGrid),
      devices(mcGrid->devices)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      SetActiveGPU forDuration(device);
      
      pld->majorantsBuffer    = rtc->createBuffer(sizeof(float));
    }
  }
  
  MCGrid::MCGrid(const DevGroup::SP &devices)
    : devices(devices)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      SetActiveGPU forDuration(device);
      
      pld->scalarRangesBuffer = rtc->createBuffer(sizeof(range1f));
    }
  }

  /* kernel CODE */
  __rtc_global
  void clearMCs(const rtc::ComputeInterface &ci, MCGrid::DD grid)
  {
    int ix = ci.getThreadIdx().x
      +ci.getBlockIdx().x*ci.getBlockDim().x;
    if (ix >= grid.dims.x*grid.dims.y*grid.dims.z) return;
    
    grid.scalarRanges[ix] = { +BARNEY_INF, -BARNEY_INF };
  }
  
  /*! re-set all cells' ranges to "infinite empty" */
  void MCGrid::clearCells()
  {
    size_t numCells = owl::common::volume(dims);
    const int bs = 1024;
    // cuda num blocks
    // const int nb = (int)divRoundUp((size_t)numCells,(size_t)bs);
    const int nb = (int)dru(numCells,bs);
    
    for (auto device : *devices) {
      auto d_grid = getDD(device);
      __rtc_launch(device->rtc,
                   clearMCs,
                   nb,bs,
                   d_grid);
    }
    for (auto device : *devices) 
      device->sync();
  }
  
    
  __rtc_global
  void mapMCs(const rtc::ComputeInterface &ci,
              MajorantsGrid::DD grid,
              TransferFunction::DD xf)
  {
    int ix = ci.getThreadIdx().x
      +ci.getBlockIdx().x*ci.getBlockDim().x;
    if (ix >= grid.dims.x*grid.dims.y*grid.dims.z) return;
    range1f scalarRange = grid.scalarRanges[ix];
    const float maj = xf.majorant(scalarRange);
    grid.majorants[ix] = maj;
  }
  
  /*! recompute all macro cells' majorant value by remap each such
    cell's value range through the given transfer function */
  void MajorantsGrid::computeMajorants(TransferFunction *xf)
  {
// #ifndef NDEBUG
//     std::cout << "-------------------------" << std::endl;
//     std::cout << "(re-)computing majorants!" << std::endl;
//     std::cout << "-------------------------" << std::endl;
// #endif
    if (dims != mcGrid->dims)
      resize(mcGrid->dims); 
    assert(xf);
    auto dims = mcGrid->dims;
    assert(dims.x > 0);
    assert(dims.y > 0);
    assert(dims.z > 0);
    size_t numCells = owl::common::volume(dims);
    const int bs = 1024;
    // cuda num blocks
    // const int nb = (int)divRoundUp((size_t)numCells,(size_t)bs);
    const int nb = (int)dru(numCells,bs);
    for (auto device : *devices) 
      device->sync();
    
    for (auto device : *devices) {
      auto d_xf = xf->getDD(device);
      auto dd = getDD(device);
      __rtc_launch(device->rtc,
                   mapMCs,
                   nb,bs,
                   dd,d_xf);
    }
    for (auto device : *devices) 
      device->sync();
  }
  
  /*! allocate memory for the given grid */
  void MajorantsGrid::resize(vec3i dims)
  {
    assert(dims.x > 0);
    assert(dims.y > 0);
    assert(dims.z > 0);
    if (dims == this->dims)
      return;
    mcGrid->resize(dims);
    this->dims = dims;
    size_t numCells = owl::common::volume(dims);
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      rtc->freeBuffer(pld->majorantsBuffer);
      
      pld->majorantsBuffer
        = rtc->createBuffer(sizeof(float)*numCells);
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
    if (dims == this->dims)
      return;
    this->dims = dims;
    size_t numCells = owl::common::volume(dims);
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      rtc->freeBuffer(pld->scalarRangesBuffer);
      pld->scalarRangesBuffer
        = rtc->createBuffer(sizeof(range1f)*numCells);
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
    
    dd.scalarRanges
      = (range1f*)pld->scalarRangesBuffer->getDD();

    dd.dims = dims;
    dd.gridOrigin = gridOrigin;
    dd.gridSpacing = gridSpacing;
    return dd;
  }

  /*! get cuda-usable device-data for given device ID (relative to
    devices in the devgroup that this gris is in */
  MajorantsGrid::DD MajorantsGrid::getDD(Device *device) 
  {
    MajorantsGrid::DD dd;

    (MCGrid::DD&)dd = mcGrid->getDD(device);
    
    PLD *pld = getPLD(device);
    dd.majorants
      = (float *)pld->majorantsBuffer->getDD();
    return dd;
  }
  
} // ::barney



