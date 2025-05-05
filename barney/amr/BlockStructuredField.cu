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

#define RTC_DEVICE_CODE 1
#include "rtcore/ComputeInterface.h"

#include "barney/amr/BlockStructuredField.h"
#include "barney/Context.h"
#include "barney/volume/MCGrid.cuh"
#include "barney/amr/BlockStructuredCUBQLSampler.h"


namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/BlockStructuredMC,
                       /*name*/BlockStructuredMC,
                       /*geomtype device data */
                       MCVolumeAccel<BlockStructuredCUBQLSampler>::DD,false,false);

  enum { MC_GRID_SIZE = 256 };

  BlockStructuredField::PLD *BlockStructuredField::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }

#if RTC_DEVICE_CODE
  __rtc_global
  void BSField_rasterGrids(const rtc::ComputeInterface &ci,
                           BlockStructuredField::DD field,
                           MCGrid::DD               grid)
  {
    const int tid = ci.launchIndex().x;
    if (tid >= field.numBlocks) return;

    const Block block = Block::getFrom(field,tid);

    vec3i numCells = block.dims;

    const box3f worldBox = getBox(field.worldBounds);
    for (int z=0;z<numCells.z;z++) {
      for (int y=0;y<numCells.y;y++) {
        for (int x=0;x<numCells.x;x++) {
          const box3f cb3 = block.cellBounds({x,y,z});
          const float scalar = block.getScalar({x,y,z});
          const box4f cellBounds(vec4f(cb3.lower,scalar),
                                 vec4f(cb3.upper,scalar));
          rasterBox(grid,worldBox,cellBounds);
        }
      }
    }
  }
#endif
  
  __rtc_global
  void computeWorldBounds(const rtc::ComputeInterface &ci,
                          box3f *pBounds,
                          const BlockStructuredField::DD dd)
  {
    int tid = ci.launchIndex().x;
    if (tid >= dd.numBlocks)
      return;

    Block block = Block::getFrom(dd,tid);
    box3f bb = block.getDomain();
    rtc::fatomicMin(&pBounds->lower.x,bb.lower.x);
    rtc::fatomicMin(&pBounds->lower.y,bb.lower.y);
    rtc::fatomicMin(&pBounds->lower.z,bb.lower.z);
    rtc::fatomicMax(&pBounds->upper.x,bb.upper.x);
    rtc::fatomicMax(&pBounds->upper.y,bb.upper.y);
    rtc::fatomicMax(&pBounds->upper.z,bb.upper.z);
  }
  
  void BlockStructuredField::buildMCs(MCGrid &grid)
  {
    // PING;
    if (grid.built()) {
      // initial grid already built
      return;
    }

    std::cout << OWL_TERMINAL_BLUE
              << "#bn.amr: building initial macro cell grid"
              << OWL_TERMINAL_DEFAULT << std::endl;
    numBlocks = perBlock.origins->count;

    // =============================================================================
    // compute macro cell grid dims
    // =============================================================================
    
    float maxWidth = reduce_max(getBox(worldBounds).size());
    vec3i dims = 1+vec3i(getBox(worldBounds).size() * ((MC_GRID_SIZE-1) / maxWidth));
    // printf("#bn.amr: chosen macro-cell dims of (%i %i %i)\n",
    //        dims.x,
    //        dims.y,
    //        dims.z);
    grid.resize(dims);
    grid.gridOrigin = worldBounds.lower;
    grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));
    
    grid.clearCells();

    for (auto device : *devices) {
#if 1
      __rtc_launch(device->rtc,
                   BSField_rasterGrids,
                   divRoundUp(numBlocks,1024),1024,
                   this->getDD(device),grid.getDD(device));
                   
                   
#else
      SetActiveGPU forDuration(device);

      BSField_mcRasterBlocks kernelData(this->getDD(device),grid.getDD(device));
      getPLD(device)->mcRasterBlocks->launch
        (divRoundUp(numBlocks,1024),1024,&kernelData);
#endif
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
    // for (auto device : *devices) {
    //   SetActiveGPU forDuration(device);
    //   PLD *pld = getPLD(device);
    //   pld->mcRasterBlocks
    //     = createCompute_BSField_mcRasterBlocks(device->rtc);
    //   pld->computeElementBBs
    //     = createCompute_BSField_computeElementBBs(device->rtc);
    // }
  }

  BlockStructuredField::DD BlockStructuredField::getDD(Device *device)
  {
    BlockStructuredField::DD dd;

    // inherited:
    (ScalarField::DD &)dd = ScalarField::getDD(device);
    
    dd.perBlock.origins = (const vec3i *)perBlock.origins->getDD(device);
    dd.perBlock.dims    = (const vec3i *)perBlock.dims->getDD(device);
    dd.perBlock.levels  = (const int   *)perBlock.levels->getDD(device);
    dd.perBlock.offsets = (const uint64_t *)perBlock.offsets->getDD(device);

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


//   struct BSField_computeElementBBs {
//     /* kernel ARGS */
//     box3f         *d_primBounds;
//     range1f       *d_primRanges;
//     BlockStructuredField::DD field;

// #if RTC_DEVICE_CODE
//     inline __rtc_device
//     void run(const rtc::ComputeInterface &ci);
// #endif
//   };

#if RTC_DEVICE_CODE
  // /* kernel FUNCTION */
  // inline __rtc_device
  // void BSField_computeElementBBs::run(const rtc::ComputeInterface &ci)
  // {
  //   const int tid = ci.launchIndex().x;
  //   if (tid >= field.numBlocks) return;

  //   Block block = Block::getFrom(field,tid);
  //   d_primBounds[tid] = block.getDomain();
  //   if (d_primRanges) d_primRanges[tid] = block.getValueRange();
  // }

  __rtc_global
  void BSField_computeElementBBs(const rtc::ComputeInterface &ci,
                                 box3f         *d_primBounds,
                                 range1f       *d_primRanges,
                                 BlockStructuredField::DD field)
  {
    const int tid = ci.launchIndex().x;
    if (tid >= field.numBlocks) return;

    Block block = Block::getFrom(field,tid);
    d_primBounds[tid] = block.getDomain();

    if (isnan(d_primBounds[tid].lower.x) ||
        isnan(d_primBounds[tid].lower.y) || 
        isnan(d_primBounds[tid].lower.z) ||
        isnan(d_primBounds[tid].upper.x) ||
        isnan(d_primBounds[tid].upper.y) || 
        isnan(d_primBounds[tid].upper.z) ) {
      printf("NAN box\n");
    }

    if (isinf(d_primBounds[tid].lower.x) ||
        isinf(d_primBounds[tid].lower.y) || 
        isinf(d_primBounds[tid].lower.z) ||
        isinf(d_primBounds[tid].upper.x) ||
        isinf(d_primBounds[tid].upper.y) || 
        isinf(d_primBounds[tid].upper.z) ) {
      printf("INF box %i - org %i %i %i dim %i %i %i cellsize %f\n",tid,
             block.origin.x,
             block.origin.y,
             block.origin.z,
             block.dims.x,
             block.dims.y,
             block.dims.z,
             block.cellSize
             );
    }
    
    if (d_primRanges) d_primRanges[tid] = block.getValueRange();
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
#if 1
    int bs = 128;
    int nb = divRoundUp(numBlocks,bs);
    __rtc_launch(device->rtc, BSField_computeElementBBs,
                 nb,bs,
                 d_primBounds,
                 d_primRanges,
                 getDD(device));
#else
    BSField_computeElementBBs args = {
      /* kernel ARGS */
      d_primBounds,
      d_primRanges,
      getDD(device)
    };
    int bs = 128;
    int nb = divRoundUp(numBlocks,bs);
    getPLD(device)->computeElementBBs->launch(nb,bs,&args);
#endif
    device->sync();
  }

  bool BlockStructuredField::setData(const std::string &member,
                                     const std::shared_ptr<Data> &value)
  {
    if (ScalarField::setData(member,value)) return true;

    if (member == "grid.origins") {
      perBlock.origins = value->as<PODData>();
      return true;
    }
    if (member == "grid.dims") {
      perBlock.dims = value->as<PODData>();
      return true;
    }
    if (member == "grid.levels") {
      perBlock.levels = value->as<PODData>();
      return true;
    }
    if (member == "grid.offsets") {
      perBlock.offsets = value->as<PODData>();
      return true;
    }
    if (member == "scalars") {
      scalars = value->as<PODData>();
      return true;
    }
    if (member == "level.refinements") {
      perLevel.refinements = value->as<PODData>();
      return true;
    }

    return false;
  }
  
  void BlockStructuredField::commit()
  {
    assert(perBlock.origins);
    assert(perBlock.dims);
    assert(perBlock.levels);
    assert(perBlock.offsets);

    numBlocks = perBlock.origins->count;
    assert(numBlocks > 0);
    assert(perBlock.dims->count == numBlocks);
    assert(perBlock.offsets->count == numBlocks);
    assert(perBlock.levels->count == numBlocks);

    // =============================================================================
    // compute world bounds
    // =============================================================================
    {
      auto dev = (*devices)[0];
      SetActiveGPU forDuration(dev);
      auto rtc = dev->rtc;
      box3f *d_worldBounds = (box3f *)rtc->allocMem(sizeof(box3f));
      rtc->copy(d_worldBounds,&worldBounds,sizeof(worldBounds));
      // PING;
      // PRINT(numBlocks);
      __rtc_launch(rtc,computeWorldBounds,
                   divRoundUp(numBlocks,128),128,
                   d_worldBounds,getDD(dev));
      rtc->sync();
      rtc->copy(&worldBounds,d_worldBounds,sizeof(worldBounds));
      // PRINT(worldBounds);
    }
    
  }

  // RTC_EXPORT_COMPUTE1D(BSField_mcRasterBlocks,BSField_mcRasterBlocks);
  // RTC_EXPORT_COMPUTE1D(BSField_computeElementBBs,BSField_computeElementBBs);
}
