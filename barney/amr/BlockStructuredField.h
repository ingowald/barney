// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Object.h"
#include "barney/ModelSlot.h"
#include "barney/geometry/IsoSurface.h"

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
    
    MCGrid::SP buildMCs() override;
    MCGrid::SP buildIsoMCs();
    MCGrid::SP getIsoMCs() override;
    MCGrid::SP getVolumeMCs() override;
    
    MCGrid::SP mcIsoGrid;
    
    /*! computes, on specified device, the array of bounding box and
        value ranges for cubql bvh consturction; one box and one value
        range per each block */
    void computeElementBBs(Device *device,
                           box3f *d_primBounds,
                           range1f *d_primRanges);
    
    VolumeAccel::SP createAccel(Volume *volume) override;

    IsoSurfaceAccel::SP createIsoAccel(IsoSurface *isoSurface) override;

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
    /*! cell-centered data extent without the half-cell BVH/query padding */
    inline __rtc_device box3f getDataDomain() const;
    inline __rtc_device bool getDataDomainContains(vec3f P) const;
    inline __rtc_device bool getDomainContains(vec3f P) const;
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
    if (!block.getDomainContains(P)) return false;

    const box3f domain = block.getDomain();
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
        if (idx_hi.x >= 0 && idx_hi.x < block.dims.x) {
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
    return sumWeights > 0.f;
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
  box3f Block::getDataDomain() const
  {
    box3f cb;
    cb.lower = vec3f(origin) * cellSize;
    cb.upper = vec3f(origin + dims) * cellSize;
    return cb;
  }

  inline __rtc_device
  bool Block::getDataDomainContains(vec3f P) const
  {
    const box3f dd = getDataDomain();
    return P.x >= dd.lower.x && P.y >= dd.lower.y && P.z >= dd.lower.z
      && P.x <  dd.upper.x && P.y <  dd.upper.y && P.z <  dd.upper.z;
  }

  inline __rtc_device
  bool Block::getDomainContains(vec3f P) const
  {
    const box3f dd = getDomain();
    return P.x >= dd.lower.x && P.y >= dd.lower.y && P.z >= dd.lower.z
      && P.x <= dd.upper.x && P.y <= dd.upper.y && P.z <= dd.upper.z;
  }

  inline __rtc_device
  int blockIndexAt(const BlockStructuredField::DD &dd, vec3f P)
  {
    int bestBlock = -1;
    int bestLevel = -1;
    for (int i = 0; i < dd.numBlocks; ++i) {
      const Block block = Block::getFrom(dd, i);
      if (!block.getDomainContains(P))
        continue;
      if (block.level > bestLevel) {
        bestLevel = block.level;
        bestBlock = i;
      }
    }
    return bestBlock;
  }

  inline __rtc_device
  int blockIndexAtCoarsest(const BlockStructuredField::DD &dd, vec3f P)
  {
    int bestBlock = -1;
    int bestLevel = 999;
    for (int i = 0; i < dd.numBlocks; ++i) {
      const Block block = Block::getFrom(dd, i);
      if (!block.getDomainContains(P))
        continue;
      if (block.level < bestLevel) {
        bestLevel = block.level;
        bestBlock = i;
      }
    }
    return bestBlock;
  }

  inline __rtc_device
  float sampleBlockStructuredAt(const BlockStructuredField::DD &dd, vec3f P)
  {
    const int bestBlock = blockIndexAt(dd, P);
    if (bestBlock < 0)
      return NAN;

    float sumWeightedValues = 0.f;
    float sumWeights = 0.f;
    dd.addBasisFunctions(sumWeightedValues, sumWeights, (uint32_t)bestBlock, P);
    return sumWeights == 0.f ? NAN : (sumWeightedValues / sumWeights);
  }

  inline __rtc_device
  float sampleBlockStructuredAtCoarsest(const BlockStructuredField::DD &dd,
                                        vec3f P)
  {
    const int bestBlock = blockIndexAtCoarsest(dd, P);
    if (bestBlock < 0)
      return NAN;

    float sumWeightedValues = 0.f;
    float sumWeights = 0.f;
    dd.addBasisFunctions(sumWeightedValues, sumWeights, (uint32_t)bestBlock, P);
    return sumWeights == 0.f ? NAN : (sumWeightedValues / sumWeights);
  }

  inline __rtc_device
  vec3f sampleBlockStructuredGrad(const BlockStructuredField::DD &dd, vec3f P)
  {
    const int bid = blockIndexAt(dd, P);
    if (bid < 0)
      return vec3f(0.f);

    const Block block = Block::getFrom(dd, bid);
    const float h = block.cellSize * 0.5f;
    const float f0 = sampleBlockStructuredAt(dd, P);
    if (isnan(f0))
      return vec3f(0.f);

    auto sampleInBlock = [&](vec3f Q) -> float {
      if (blockIndexAt(dd, Q) != bid)
        return NAN;
      return sampleBlockStructuredAt(dd, Q);
    };

    auto deriv = [&](const vec3f &axis) -> float {
      const float fp = sampleInBlock(P + h * axis);
      const float fm = sampleInBlock(P - h * axis);
      if (!isnan(fp) && !isnan(fm))
        return (fp - fm) / (2.f * h);
      if (!isnan(fp))
        return (fp - f0) / h;
      if (!isnan(fm))
        return (f0 - fm) / h;
      return 0.f;
    };

    return vec3f(deriv(vec3f(1.f, 0.f, 0.f)),
                 deriv(vec3f(0.f, 1.f, 0.f)),
                 deriv(vec3f(0.f, 0.f, 1.f)));
  }

  inline __rtc_device
  vec3f sampleBlockStructuredGradCoarsest(const BlockStructuredField::DD &dd,
                                          vec3f P)
  {
    const int bid = blockIndexAtCoarsest(dd, P);
    if (bid < 0)
      return vec3f(0.f);

    const Block block = Block::getFrom(dd, bid);
    const float h = block.cellSize * 0.5f;
    const float f0 = sampleBlockStructuredAtCoarsest(dd, P);
    if (isnan(f0))
      return vec3f(0.f);

    auto sampleInBlock = [&](vec3f Q) -> float {
      if (blockIndexAtCoarsest(dd, Q) != bid)
        return NAN;
      return sampleBlockStructuredAtCoarsest(dd, Q);
    };

    auto deriv = [&](const vec3f &axis) -> float {
      const float fp = sampleInBlock(P + h * axis);
      const float fm = sampleInBlock(P - h * axis);
      if (!isnan(fp) && !isnan(fm))
        return (fp - fm) / (2.f * h);
      if (!isnan(fp))
        return (fp - f0) / h;
      if (!isnan(fm))
        return (f0 - fm) / h;
      return 0.f;
    };

    return vec3f(deriv(vec3f(1.f, 0.f, 0.f)),
                 deriv(vec3f(0.f, 1.f, 0.f)),
                 deriv(vec3f(0.f, 0.f, 1.f)));
  }

  template<typename T>
  inline __rtc_device int amrBlockAt(const T &, vec3f)
  { return -1; }

  template<typename T>
  inline __rtc_device bool amrNearDataDomainFace(const T &, vec3f)
  { return false; }

  template<typename T>
  inline __rtc_device vec3f amrSampleGrad(const T &, vec3f)
  { return vec3f(0.f); }

  inline __rtc_device int amrBlockAt(const BlockStructuredField::DD &dd,
                                     vec3f P)
  { return blockIndexAt(dd, P); }

  inline __rtc_device vec3f amrSampleGrad(const BlockStructuredField::DD &dd,
                                          vec3f P)
  { return sampleBlockStructuredGrad(dd, P); }

  inline __rtc_device
  bool amrHitIsSeamArtifact(const BlockStructuredField::DD &dd,
                            vec3f P,
                            float isoValue)
  {
    const int bid = blockIndexAt(dd, P);
    if (bid < 0)
      return true;

    const Block block = Block::getFrom(dd, bid);
    const box3f domain = block.getDataDomain();
    const float eps = block.cellSize * 0.05f;
    const float fP = sampleBlockStructuredAt(dd, P);
    if (isnan(fP))
      return true;

    const float dist[3] = {
      min(P.x - domain.lower.x, domain.upper.x - P.x),
      min(P.y - domain.lower.y, domain.upper.y - P.y),
      min(P.z - domain.lower.z, domain.upper.z - P.z),
    };

    for (int axis = 0; axis < 3; ++axis) {
      if (dist[axis] > eps)
        continue;
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        vec3f Q = P;
        Q[axis] += float(sgn) * 2.f * eps;
        const int other = blockIndexAt(dd, Q);
        if (other < 0 || other == bid)
          continue;
        const float fQ = sampleBlockStructuredAt(dd, Q);
        if (isnan(fQ))
          continue;
        const float jump = fabsf(fQ - fP);
        const float toIso
          = fminf(fabsf(fP - isoValue), fabsf(fQ - isoValue));
        if (jump > 3.f * toIso + 1e-4f)
          return true;
      }
    }
    return false;
  }

  template<typename T>
  inline __rtc_device bool amrHitIsSeamArtifact(const T &, vec3f, float)
  { return false; }

  inline __rtc_device
  Block Block::getFrom(const BlockStructuredField::DD &dd, int blockID, bool dbg)
  {
    Block block;
    block.origin   = dd.perBlock.origins[blockID];
    block.dims     = dd.perBlock.dims[blockID];
    block.level    = dd.perBlock.levels[blockID];
    block.cellSize = powf(2.f, -(float)block.level);
    block.scalars  = dd.scalars+dd.perBlock.offsets[blockID];
    return block;
  }
#endif
}
