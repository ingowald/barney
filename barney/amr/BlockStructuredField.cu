// ======================================================================== //
// Copyright 2023++ Ingo Wald                                               //
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

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/BlockStructuredMC,
                       /*name*/BlockStructuredMC,
                       /*geomtype device data */
                       MCVolumeAccel<BlockStructuredCUBQLSampler>::DD,false,false);
                       // MCVolumeAccel<BlockStructuredSampler>::DD,false,false);
  RTC_IMPORT_COMPUTE1D(BSField_mcRasterBlocks);
  RTC_IMPORT_COMPUTE1D(BSField_computeElementBBs);

  enum { MC_GRID_SIZE = 128 };

  BlockStructuredField::PLD *BlockStructuredField::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }

  /*! block structured field, rastering its blocks into a macro-cell grid */
  struct BSField_mcRasterBlocks {
    /* kernel data */
    BlockStructuredField::DD field;
    MCGrid::DD               grid;

#if RTC_DEVICE_CODE
  /*! block structured field, rastering its blocks into a macro-cell grid */
    inline __rtc_device
    void run(const rtc::ComputeInterface &ci);
#endif
  };


#if RTC_DEVICE_CODE
  /*! block structured field, rastering its blocks into a macro-cell grid */
  inline __rtc_device
  void BSField_mcRasterBlocks::run(const rtc::ComputeInterface &ci)
  {
    const int tIdx = ci.getBlockIdx().x*ci.getBlockDim().x + ci.getThreadIdx().x;
    if (tIdx >= field.numBlocks) return;

    const Block block = Block::getFrom(field,tIdx);

    vec3i numCells = block.dims;

    for (int z=0;z<numCells.z;z++) {
      for (int y=0;y<numCells.y;y++) {
        for (int x=0;x<numCells.x;x++) {
          const box3f cb3 = block.cellBounds({x,y,z});
          const float scalar = block.getScalar({x,y,z});
          const box4f cellBounds(vec4f(cb3.lower,scalar),
                                 vec4f(cb3.upper,scalar));
          rasterBox(grid,getBox(field.worldBounds),cellBounds);
        }
      }
    }
  }
#endif

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

    int numBlocks = perBlock.origins->count;
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);

      BSField_mcRasterBlocks kernelData(this->getDD(device),grid.getDD(device));
      getPLD(device)->mcRasterBlocks->launch
        (divRoundUp(numBlocks,1024),1024,&kernelData);
    }
    
    for (auto device : *devices) 
      device->sync();
  }
  

  BlockStructuredField::~BlockStructuredField()
  {}
    

  
  BlockStructuredField::BlockStructuredField(Context *context,
                                             const DevGroup::SP &devices)
    : ScalarField(context,devices)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      PLD *pld = getPLD(device);
      pld->mcRasterBlocks
        = createCompute_BSField_mcRasterBlocks(device->rtc);
      pld->computeElementBBs
        = createCompute_BSField_computeElementBBs(device->rtc);
    }
  }

  BlockStructuredField::DD BlockStructuredField::getDD(Device *device)
  {
    BlockStructuredField::DD dd;

    dd.perBlock.origins = (const vec3i *)perBlock.origins->getDD(device);
    dd.perBlock.dims    = (const vec3i *)perBlock.dims->getDD(device);
    dd.perBlock.levels  = (const int   *)perBlock.levels->getDD(device);
    dd.perBlock.offsets = (const int   *)perBlock.offsets->getDD(device);

    dd.perLevel.refinements = (const int *)perLevel.refinements->getDD(device);
    
    dd.scalars      = (const float *)scalars->getDD(device);
    dd.numBlocks    = perBlock.origins->count;

    return dd;
  }

  VolumeAccel::SP BlockStructuredField::createAccel(Volume *volume)
  {
    auto sampler
      = std::make_shared<BlockStructuredCUBQLSampler>(this);
    return std::make_shared<MCVolumeAccel<BlockStructuredCUBQLSampler>>
      (volume,
       createGeomType_BlockStructuredMC,
       sampler);
  }


  struct BSField_computeElementBBs {
    /* kernel ARGS */
    box3f         *d_primBounds;
    range1f       *d_primRanges;
    BlockStructuredField::DD field;

#if RTC_DEVICE_CODE
    inline __rtc_device
    void run(const rtc::ComputeInterface &ci);
#endif
  };

#if RTC_DEVICE_CODE
  /* kernel FUNCTION */
  inline __rtc_device
  void BSField_computeElementBBs::run(const rtc::ComputeInterface &ci)
  {
    const int tid = ci.launchIndex().x;
    if (tid >= field.numBlocks) return;

    Block block = Block::getFrom(field,tid);
    d_primBounds[tid] = block.getDomain();
    if (d_primRanges) printf("value ranges not implemented for bsfield");
  }
#endif
    
  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitmives ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeaable on specified device */
  void BlockStructuredField::computeElementBBs(Device  *device,
                                               box3f   *d_primBounds,
                                               range1f *d_primRanges)
  {
    BSField_computeElementBBs args = {
      /* kernel ARGS */
      d_primBounds,
      d_primRanges,
      getDD(device)
    };
    int bs = 128;
    int nb = divRoundUp(numBlocks,bs);
    getPLD(device)->computeElementBBs->launch(nb,bs,&args);
    device->sync();
  }

  RTC_EXPORT_COMPUTE1D(BSField_mcRasterBlocks,BSField_mcRasterBlocks);
  RTC_EXPORT_COMPUTE1D(BSField_computeElementBBs,BSField_computeElementBBs);
}
