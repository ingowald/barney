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

  MCGrid::MCGrid(DevGroup *devGroup)
    : devGroup(devGroup)
  {
    assert(devGroup);
    scalarRangesBuffer
      = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT2,1,nullptr);
    majorantsBuffer
      = owlDeviceBufferCreate(devGroup->owl,OWL_FLOAT,1,nullptr);
  }
                            
  __global__ void g_clearMCs(MCGrid::DD grid)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= grid.dims.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= grid.dims.y) return;
    int iz = threadIdx.z+blockIdx.z*blockDim.z; if (iz >= grid.dims.z) return;
    
    int ii = ix + grid.dims.x*(iy + grid.dims.y*(iz));
    grid.scalarRanges[ii] = { +INFINITY, -INFINITY };
  }
  
  /*! re-set all cells' ranges to "infinite empty" */
  void MCGrid::clearCells()
  {
    const vec3i bs = 4;
    const vec3i nb = divRoundUp(dims,bs);
    for (auto dev : devGroup->devices) {
      SetActiveGPU forDuration(dev);
      BARNEY_CUDA_SYNC_CHECK();
      auto d_grid = getDD(dev->owlID);
      g_clearMCs
        <<<(dim3)nb,(dim3)bs>>>
        (d_grid);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }
  
  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  __global__ void mapMacroCells(MCGrid::DD grid,
                                TransferFunction::DD xf)
  {
    vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
               threadIdx.y+blockIdx.y*blockDim.y,
               threadIdx.z+blockIdx.z*blockDim.z);

    if (mcID.x >= grid.dims.x) return;
    if (mcID.y >= grid.dims.y) return;
    if (mcID.z >= grid.dims.z) return;
    
    int mcIdx = mcID.x + grid.dims.x*(mcID.y + grid.dims.y*mcID.z);
    range1f scalarRange = grid.scalarRanges[mcIdx];
    grid.majorants[mcIdx] = xf.majorant(scalarRange);
  }
  
  /*! recompute all macro cells' majorant value by remap each such
    cell's value range through the given transfer function */
  void MCGrid::computeMajorants(const TransferFunction *xf)
  {
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "RECOMPUTING MAJORANTS" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    assert(xf);
    assert(dims.x > 0);
    assert(dims.y > 0);
    assert(dims.z > 0);
    const vec3i bs = 4;
    // cuda num blocks
    const vec3i nb = divRoundUp(dims,bs);

    PRINT(dims);
    
    for (auto dev : xf->devGroup->devices) {
      BARNEY_CUDA_SYNC_CHECK();
      // for (int devID=0;devID<xf->devGroup->size();devID++) {
      SetActiveGPU forDuration(dev);
      auto d_xf = xf->getDD(dev->owlID);
      PRINT(d_xf.values);
      PRINT(d_xf.domain);
      mapMacroCells
        <<<(dim3)nb,(dim3)bs>>>
        (getDD(dev->owlID),d_xf);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }

  /*! allocate memory for the given grid */
  void MCGrid::resize(vec3i dims)
  {
    assert(dims.x > 0);
    assert(dims.y > 0);
    assert(dims.z > 0);
    std::cout << "resizing grid to " << dims << std::endl;
    this->dims = dims;
    int numCells = owl::common::volume(dims);
    PRINT(numCells);
    owlBufferResize(majorantsBuffer,numCells);
    owlBufferResize(scalarRangesBuffer,numCells);
  }
    
  
  /*! get cuda-usable device-data for given device ID (relative to
    devices in the devgroup that this gris is in */
  MCGrid::DD MCGrid::getDD(int devID) const
  {
    MCGrid::DD dd;
    
    assert(majorantsBuffer);
    dd.majorants
      = (float *)owlBufferGetPointer(majorantsBuffer,devID);

    assert(scalarRangesBuffer);
    dd.scalarRanges
      = (range1f*)owlBufferGetPointer(scalarRangesBuffer,devID);
    
    dd.dims = dims;
    
    return dd;
  }

  void MCGrid::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back
      ({"majorants",OWL_BUFPTR,base+OWL_OFFSETOF(DD,majorants)});
    vars.push_back
      ({"scalarRanges",OWL_BUFPTR,base+OWL_OFFSETOF(DD,scalarRanges)});
    vars.push_back
      ({"dims",OWL_INT3,base+OWL_OFFSETOF(DD,dims)});
    vars.push_back
      ({"gridOrigin",OWL_FLOAT3,base+OWL_OFFSETOF(DD,gridOrigin)});
    vars.push_back
      ({"gridSpacing",OWL_FLOAT3,base+OWL_OFFSETOF(DD,gridSpacing)});
  }
  
  void MCGrid::setVariables(OWLGeom geom)
  {
    owlGeomSetBuffer(geom,"majorants",majorantsBuffer);
    owlGeomSet3i(geom,"dims",dims.x,dims.y,dims.z);
    owlGeomSet3f(geom,"gridOrigin",gridOrigin.x,gridOrigin.y,gridOrigin.z);
    owlGeomSet3f(geom,"gridSpacing",gridSpacing.x,gridSpacing.y,gridSpacing.z);
  }
  
} // ::vopat
