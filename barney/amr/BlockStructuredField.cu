// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/amr/BlockStructuredField.h"
#include "barney/Context.h"
#include "barney/volume/MCGrid.cuh"
#include "barney/amr/BlockStructuredCUBQLSampler.h"

#define BUFFER_CREATE owlDeviceBufferCreate
// #define BUFFER_CREATE owlManagedMemoryBufferCreate

namespace barney {

  extern "C" char BlockStructuredMC_ptx[];

  enum { MC_GRID_SIZE = 128 };

  __global__ void rasterBlocks(MCGrid::DD grid,
                               BlockStructuredField::DD field)
  {
    const int tIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (tIdx > field.numBlocks) return;

    const BlockStructuredField::Block &blk = field.getBlock(tIdx);

    vec3i numCells = blk.numCells();

    auto linearIndex = [numCells](const int x, const int y, const int z) {
                         return z*numCells.y*numCells.x + y*numCells.x + x;
                       };

    for (int z=0;z<numCells.z;z++) {
      for (int y=0;y<numCells.y;y++) {
        for (int x=0;x<numCells.x;x++) {
          const box3f cb3 = blk.cellBounds(vec3i(x,y,z));
          const float scalar = field.blockScalars[blk.scalarOffset + linearIndex(x,y,z)];
          const box4f cellBounds(vec4f(cb3.lower,scalar),
                                 vec4f(cb3.upper,scalar));
          rasterBox(grid,getBox(field.worldBounds),cellBounds);
        }
      }
    }
  }

  void BlockStructuredField::buildMCs(MCGrid &grid)
  {
    if (grid.built()) {
      // initial grid already built
      return;
    }
    
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.amr: building initial macro cell grid"
              << OWL_TERMINAL_DEFAULT << std::endl;

    float maxWidth = reduce_max(getBox(worldBounds).size());
    vec3i dims = 1+vec3i(getBox(worldBounds).size() * ((MC_GRID_SIZE-1) / maxWidth));
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
    for (auto dev : devGroup->devices) {
      SetActiveGPU forDuration(dev);
      auto d_field = getDD(dev->owlID);
      auto d_grid = grid.getDD(dev->owlID);
      rasterBlocks
        <<<divRoundUp(int(blockIDs.size()),1024),1024>>>
        (d_grid,d_field);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }


  /*! computes - ON CURRENT DEVICE - the given mesh's block filter domains
      and per-block scalar ranges, and writes those into givne
      pre-allocated device mem location */
  __global__
  void g_computeBlockFilterDomains(box3f *d_primBounds,
                                   range1f *d_primRanges,
                                   BlockStructuredField::DD field)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= field.numBlocks) return;

    auto block = field.getBlock(tid);
    box4f eb = block.filterDomain();
    d_primBounds[tid] = getBox(eb);
    if (d_primRanges) d_primRanges[tid] = getRange(eb);
  }

  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitmives ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeaable on specified device */
  void BlockStructuredField::computeBlockFilterDomains(int deviceID,
                                                       box3f *d_primBounds,
                                                       range1f *d_primRanges)
  {
    SetActiveGPU forDuration(devGroup->devices[deviceID]);
    int bs = 1024;
    int nb = divRoundUp(int(blockIDs.size()),bs);
    g_computeBlockFilterDomains
      <<<nb,bs>>>(d_primBounds,d_primRanges,getDD(deviceID));
    BARNEY_CUDA_SYNC_CHECK();
  }

  BlockStructuredField::BlockStructuredField(DataGroup *owner,
                                             std::vector<box3i> &_blockBounds,
                                             std::vector<int> &_blockLevels,
                                             std::vector<int> &_blockOffsets,
                                             std::vector<float> &_blockScalars)
    : ScalarField(owner),
      blockBounds(std::move(_blockBounds)),
      blockLevels(std::move(_blockLevels)),
      blockOffsets(std::move(_blockOffsets)),
      blockScalars(std::move(_blockScalars))
  {
    size_t numBlocks = blockBounds.size();
    blockIDs.resize(numBlocks);
    valueRanges.resize(numBlocks);

    for (size_t blockID=0;blockID<numBlocks;++blockID) {
      Block block;
      block.ID = (int)blockID;
      block.bounds = blockBounds[blockID];
      block.level = blockLevels[blockID];
      block.scalarOffset = blockOffsets[blockID];
      block.valueRange = range1f();

      int numScalars = block.numCells().x*block.numCells().y*block.numCells().z;
      for (int s=0;s<numScalars;++s) {
        float f = blockScalars[block.scalarOffset+s];
        block.valueRange.extend(f);
      }

      blockIDs[blockID] = block.ID;
      valueRanges[blockID] = block.valueRange;
      worldBounds.extend(getBox(block.worldBounds()));
    }
    PRINT(worldBounds);
    assert(!valueRanges.empty());
    assert(!blockIDs.empty());

    blockBoundsBuffer
      = BUFFER_CREATE(getOWL(),
                              OWL_USER_TYPE(box3i),
                              blockBounds.size(),
                              blockBounds.data());

    blockLevelsBuffer
      = BUFFER_CREATE(getOWL(),
                              OWL_INT,
                              blockLevels.size(),
                              blockLevels.data());

    blockOffsetsBuffer
      = BUFFER_CREATE(getOWL(),
                              OWL_INT,
                              blockOffsets.size(),
                              blockOffsets.data());

    blockScalarsBuffer
      = BUFFER_CREATE(getOWL(),
                              OWL_FLOAT,
                              blockScalars.size(),
                              blockScalars.data());

    blockIDsBuffer
      = BUFFER_CREATE(getOWL(),
                              OWL_UINT,
                              blockIDs.size(),
                              blockIDs.data());
    valueRangesBuffer
      = BUFFER_CREATE(getOWL(),
                              OWL_USER_TYPE(range1f),
                              valueRanges.size(),
                              valueRanges.data());
  }

  BlockStructuredField::DD BlockStructuredField::getDD(int devID)
  {
    BlockStructuredField::DD dd;

    dd.blockBounds  = (const box3i    *)owlBufferGetPointer(blockBoundsBuffer,devID);
    dd.blockLevels  = (const int      *)owlBufferGetPointer(blockLevelsBuffer,devID);
    dd.blockOffsets = (const int      *)owlBufferGetPointer(blockOffsetsBuffer,devID);
    dd.blockScalars = (const float    *)owlBufferGetPointer(blockScalarsBuffer,devID);
    dd.blockIDs     = (const uint32_t *)owlBufferGetPointer(blockIDsBuffer,devID);
    dd.valueRanges  = (const range1f  *)owlBufferGetPointer(valueRangesBuffer,devID);
    dd.numBlocks    = (int)blockIDs.size();
    dd.worldBounds  = worldBounds;

    return dd;
  }

  ScalarField *DataGroup::createBlockStructuredAMR(std::vector<box3i> &blockBounds,
                                                   std::vector<int> &blockLevels,
                                                   std::vector<int> &blockOffsets,
                                                   std::vector<float> &blockScalars)
  {
    ScalarField::SP sf
      = std::make_shared<BlockStructuredField>(this,
                                               blockBounds,
                                               blockLevels,
                                               blockOffsets,
                                               blockScalars);
    return getContext()->initReference(sf);
  }

  void BlockStructuredField::setVariables(OWLGeom geom)
  {
    ScalarField::setVariables(geom);
    
    // ------------------------------------------------------------------
    owlGeomSetBuffer(geom,"field.blockBounds",blockBoundsBuffer);
    owlGeomSetBuffer(geom,"field.blockLevels",blockLevelsBuffer);
    owlGeomSetBuffer(geom,"field.blockOffsets",blockOffsetsBuffer);
    owlGeomSetBuffer(geom,"field.blockScalars",blockScalarsBuffer);
    owlGeomSetBuffer(geom,"field.blockIDs",blockIDsBuffer);
    owlGeomSetBuffer(geom,"field.valueRanges",valueRangesBuffer);
  }
  
  void BlockStructuredField::DD::addVars(std::vector<OWLVarDecl> &vars, int myOfs)
  {
    ScalarField::DD::addVars(vars,myOfs);
    std::vector<OWLVarDecl> mine = 
      {
       { "field.blockBounds",    OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,blockBounds) },
       { "field.blockLevels",    OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,blockLevels) },
       { "field.blockOffsets",   OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,blockOffsets) },
       { "field.blockScalars",   OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,blockScalars) },
       { "field.blockIDs",       OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,blockIDs) },
       { "field.valueRanges",    OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,valueRanges) },
      };
    for (auto var : mine)
      vars.push_back(var);
  }

  VolumeAccel::SP BlockStructuredField::createAccel(Volume *volume)
  {
    const char *methodFromEnv = getenv("BARNEY_AMR");
    std::string method = (methodFromEnv ? methodFromEnv : "DDA");

    if (method == "DDA" || method == "MCDDA")
      return std::make_shared<MCDDAVolumeAccel<BlockStructuredCUBQLSampler>::Host>
        (this,volume,BlockStructuredMC_ptx);

    if (method == "MCRTX")
      return std::make_shared<MCRTXVolumeAccel<BlockStructuredCUBQLSampler>::Host>
        (this,volume,BlockStructuredMC_ptx);
    
    throw std::runtime_error("unknown BARNEY_AMR accelerator method");
  }
}
