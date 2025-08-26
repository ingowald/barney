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

#pragma once

#include "barney/Object.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {

  struct Block;
  
  struct BlockStructuredField : public ScalarField
  {
    typedef std::shared_ptr<BlockStructuredField> SP;

    struct PLD {
      // rtc::ComputeKernel1D *mcRasterBlocks = 0;
      // rtc::ComputeKernel1D *computeElementBBs = 0;
      Block *blocks;
      float *scalars;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
    
    struct DD : public ScalarField::DD {

#if RTC_DEVICE_CODE
      /* compute basis function contribution of given block at point P, and add
         that to 'sumWeightedValues' and 'sumWeights'. returns true if P is
         inside the block *filter domain*, false if outside (in which case the
         out params are not defined) */
      inline __rtc_device bool addBasisFunctions(float &sumWeightedValues,
                                             float &sumWeights,
                                             uint32_t bid,
                                             vec3f P) const;
#endif
      const float   *scalars;
      struct {
        const vec3i *origins;
        const vec3i *dims;
        const int   *levels;
        const uint64_t *offsets;
      } perBlock;
      struct {
        const int   *refinements;
      } perLevel;
      int numBlocks;
    };

    BlockStructuredField(Context *context,
                         const DevGroup::SP &devices);
    virtual ~BlockStructuredField() override;
    
    DD getDD(Device *device);
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setData(const std::string &member,
                 const std::shared_ptr<Data> &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    void buildMCs(MCGrid &macroCells) override;
    
    /*! computes, on specified device, the array of bounding box and
        value ranges for cubql bvh consturction; one box and one value
        range per each block */
    void computeElementBBs(Device *device,
                           box3f *d_primBounds,
                           range1f *d_primRanges);
    
    VolumeAccel::SP createAccel(Volume *volume) override;

    struct {
      PODData::SP/*3i*/ origins    = 0;
      PODData::SP/*3i*/ dims       = 0;
      PODData::SP/*1i*/ levels     = 0;
      PODData::SP/*1l*/ offsets    = 0;
    } perBlock;
    struct {
      PODData::SP/*1i*/ refinements = 0;
    } perLevel;
    PODData::SP/*1f*/   scalars     = 0;
    int                 numBlocks   = 0;
  };


  struct Block
  {
#if RTC_DEVICE_CODE
    static
    inline __rtc_device Block getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg=false);
    
    inline __rtc_device float getScalar(const vec3i cellID) const;
    inline __rtc_device box3f cellBounds(const vec3i cellID) const;
    inline __rtc_device box3f getDomain() const;
    inline __rtc_device range1f getValueRange() const;
#endif
    vec3i origin;
    vec3i dims;
    int   level;
    float cellSize;
    const float *scalars;
  };
  
  
#if RTC_DEVICE_CODE
  /* compute basis function contribution of given block at point P, and add
     that to 'sumWeightedValues' and 'sumWeights'. returns true if P is inside
     the block *filter domain*, false if outside (in which case the out params
     are not defined) */
  inline __rtc_device
  bool BlockStructuredField::DD::addBasisFunctions(float &sumWeightedValues,
                                                   float &sumWeights,
                                                   uint32_t bid,
                                                   vec3f P) const
  {
    const auto block = Block::getFrom(*this,bid);
    const box3f domain = block.getDomain();

    if (!domain.contains(P)) return false;

    const vec3f cellCenter000 = domain.lower+vec3f(block.cellSize);
    const vec3f localPos
      = (P-cellCenter000) / block.cellSize;
    
    vec3f floor_localPos(floorf(localPos.x),
                         floorf(localPos.y),
                         floorf(localPos.z));
    vec3i idx_lo   = vec3i(floor_localPos);
    idx_lo = max(vec3i(-1), idx_lo);
    const vec3i idx_hi   = idx_lo + vec3i(1);
    const vec3f frac     = localPos - floor_localPos;
    const vec3f neg_frac = vec3f(1.f) - frac;

    if (idx_lo.z >= 0 && idx_lo.z < block.dims.z) {
      if (idx_lo.y >= 0 && idx_lo.y < block.dims.y) {
        if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
          const float scalar = block.getScalar({idx_lo.x,idx_lo.y,idx_lo.z});
          const float weight = (neg_frac.z)*(neg_frac.y)*(neg_frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
        if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
          const float scalar = block.getScalar({idx_hi.x,idx_lo.y,idx_lo.z});
          const float weight = (neg_frac.z)*(neg_frac.y)*(frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
      }
      if (idx_hi.y >= 0 && idx_hi.y < block.dims.y) {
        if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
          const float scalar = block.getScalar({idx_lo.x,idx_hi.y,idx_lo.z});
          const float weight = (neg_frac.z)*(frac.y)*(neg_frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
        if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
          const float scalar = block.getScalar({idx_hi.x,idx_hi.y,idx_lo.z});
          const float weight = (neg_frac.z)*(frac.y)*(frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
      }
    }
        
    if (idx_hi.z >= 0 && idx_hi.z < block.dims.z) {
      if (idx_lo.y >= 0 && idx_lo.y < block.dims.y) {
        if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
          const float scalar = block.getScalar({idx_lo.x,idx_lo.y,idx_hi.z});
          const float weight = (frac.z)*(neg_frac.y)*(neg_frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
        if (idx_hi.x < block.dims.x) {
          const float scalar = block.getScalar({idx_hi.x,idx_lo.y,idx_hi.z});
          const float weight = (frac.z)*(neg_frac.y)*(frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
      }
      if (idx_hi.y >= 0 && idx_hi.y < block.dims.y) {
        if (idx_lo.x >= 0 && idx_lo.x < block.dims.x) {
          const float scalar = block.getScalar({idx_lo.x,idx_hi.y,idx_hi.z});
          const float weight = (frac.z)*(frac.y)*(neg_frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
        if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
          const float scalar = block.getScalar({idx_hi.x,idx_hi.y,idx_hi.z});
          const float weight = (frac.z)*(frac.y)*(frac.x);
          sumWeights += weight;
          sumWeightedValues += weight*scalar;
        }
      }
    }
    return true;
  }

  inline __rtc_device
  float Block::getScalar(const vec3i cellID) const
  {
    const int idx
      = 
      + cellID.x
      + cellID.y * dims.x
      + cellID.z * dims.x*dims.y;
    return scalars[idx];
  }

  inline __rtc_device
  box3f Block::cellBounds(const vec3i cellID) const
  {
    box3f cb;
    cb.lower = (vec3f(origin+cellID)-.5f)*cellSize;
    cb.upper = cb.lower + 2.f*cellSize;
    return cb;
  }

  inline __rtc_device
  range1f Block::getValueRange() const
  {
    range1f range;
    for (int i=0;i<dims.x*dims.y*dims.z;i++)
      range.extend(scalars[i]);
    return range;
  }
  
  inline __rtc_device
  box3f Block::getDomain() const
  {
    box3f cb;
    cb.lower = (vec3f(origin)-.5f)*cellSize;
    cb.upper = (vec3f(origin+dims)+.5f)*cellSize;
    return cb;
  }

  inline __rtc_device
  Block Block::getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg)
  {
    // if (blockID == 13000) dbg = true;
    Block block;
    block.origin   = dd.perBlock.origins[blockID];
    block.dims     = dd.perBlock.dims[blockID];
    block.level    = dd.perBlock.levels[blockID];
    block.cellSize = (powf(dd.perLevel.refinements[block.level], block.level));
    // printf("offset %li\n",dd.perBlock.offsets[blockID]);
    block.scalars  = dd.scalars+dd.perBlock.offsets[blockID];

    // dbg = blockID < 10 || blockID >= 227200;
    // if (dbg) {
    //   box3f dom = block.getDomain();
    //   printf("dom (%f %f %f) (%f %f %f) cs %f\n",
    //          dom.lower.x,
    //          dom.lower.y,
    //          dom.lower.z,
    //          dom.upper.x,
    //          dom.upper.y,
    //          dom.upper.z,
    //          block.cellSize);
    // }
    return block;
  }
#endif
}
