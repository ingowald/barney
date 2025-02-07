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

#include "barney/nanovdb/NanoVDBField.h"
#include "barney/Context.h"
#include "barney/volume/MCAccelerator.h"
#include "barney/volume/MCGrid.cuh"

namespace barney {

  extern "C" char NanoVDBMC_ptx[];

  enum { MC_GRID_SIZE = 16 };

  __global__ void computeMCs(MCGrid::DD grid,
                             NanoVDBField::DD field)
  {
    vec3i mcID = vec3i(threadIdx) + vec3i(blockIdx) * vec3i(blockDim);
    if (mcID.x >= grid.dims.x) return;
    if (mcID.y >= grid.dims.y) return;
    if (mcID.z >= grid.dims.z) return;

    const box3f mcBounds(grid.gridOrigin+vec3f(mcID)*grid.gridSpacing,
                         grid.gridOrigin+vec3f(mcID+1)*grid.gridSpacing);

    nanovdb::CoordBBox bbox(
        {(int)mcBounds.lower.x,(int)mcBounds.lower.y,(int)mcBounds.lower.z},
        {(int)mcBounds.upper.x+1,(int)mcBounds.upper.y+1,(int)mcBounds.upper.z+1});

    auto acc = field.gridPtr->getAccessor();

    range1f scalarRange;
    for (nanovdb::CoordBBox::Iterator iter = bbox.begin(); iter; ++iter) {
      float value = acc.getValue(*iter);
      scalarRange.extend(value);
    }

    const box4f cellBounds(vec4f(mcBounds.lower,scalarRange.lower),
                           vec4f(mcBounds.upper,scalarRange.upper));
    rasterBox(grid,getBox(field.worldBounds),cellBounds);
  }

  NanoVDBField::NanoVDBField(Context *context, int slot,
                             std::vector<float> &gridData)
    : ScalarField(context,slot)
  {
    for (auto dev : getDevices()) {
      SetActiveGPU forDuration(dev);
      size_t sizeInBytes = gridData.size() * sizeof(gridData[0]);
      printf("NanoVDBField::NanoVDBField: %lld, %lld\n", gridData.size(), sizeof(gridData[0]));

      int devID = dev->owlID;

      cudaStream_t stream;
      cudaStreamCreate(&stream);
      nanovdb::cuda::DeviceBuffer buffer(sizeInBytes,
                                         /*host:*/true,
                                         &stream);
      memcpy(buffer.data(), gridData.data(), sizeInBytes);
      gridHandles.emplace_back();
      gridHandles[devID] = std::move(buffer);

      gridHandles[devID].deviceUpload(stream, false);

      cudaStreamDestroy(stream);
    }

    auto bbox = gridHandles[0].gridMetaData()->indexBBox();

    auto lower = bbox.min();
    auto upper = bbox.max();

    worldBounds.extend(box3f({(float)lower[0],(float)lower[1],(float)lower[2]},
                             {(float)upper[0],(float)upper[1],(float)upper[2]}));
  }

  void NanoVDBField::setVariables(OWLGeom geom)
  {
    ScalarField::setVariables(geom);

    // ------------------------------------------------------------------
    for (auto dev : getDevices()) {
      auto gridPtr = getDD(dev).gridPtr;
      owlGeomSetRaw(geom,"gridPtr",&gridPtr,dev->owlID);
    }
  }

  void NanoVDBField::buildMCs(MCGrid &grid)
  {

    if (grid.built()) {
      // initial grid already built
      return;
    }
    
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.nanovdb: building initial macro cell grid"
              << OWL_TERMINAL_DEFAULT << std::endl;

    vec3i dims = MC_GRID_SIZE;
    printf("#bn.amr: chosen macro-cell dims of (%i %i %i)\n",
           dims.x,
           dims.y,
           dims.z);
    grid.resize(dims);
    grid.gridOrigin = worldBounds.lower;
    grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));
    
    grid.clearCells();
    const vec3i bs = 4;
    const vec3i nb = divRoundUp(dims,bs);
    for (auto dev : getDevices()) {
      SetActiveGPU forDuration(dev);
      auto d_field = getDD(dev);
      auto d_grid = grid.getDD(dev);
      CHECK_CUDA_LAUNCH(computeMCs,
                        (const dim3&)nb,(const dim3&)bs,0,0,
                        //
                        d_grid,d_field);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }

  NanoVDBField::DD NanoVDBField::getDD(const Device::SP &device)
  {
    NanoVDBField::DD dd;
    int devID = device->owlID;

    dd.gridPtr = gridHandles[devID].deviceGrid<float>();
    dd.worldBounds = worldBounds;

    return dd;
  }

  VolumeAccel::SP NanoVDBField::createAccel(Volume *volume)
  {
    return std::make_shared<MCDDAVolumeAccel<NanoVDBSampler>::Host>
      (this,volume,NanoVDBMC_ptx);
  }

  void NanoVDBField::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    ScalarField::DD::addVars(vars,base);
    vars.push_back
      ({"gridPtr",OWL_USER_TYPE(nanovdb::NanoGrid<float> *),base+OWL_OFFSETOF(DD,gridPtr)});
  }
}
