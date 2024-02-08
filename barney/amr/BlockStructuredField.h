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

#pragma once

#include "barney/DataGroup.h"

namespace barney {

  struct BlockStructuredField : public ScalarField
  {
    typedef std::shared_ptr<BlockStructuredField> SP;

    // /*! returns part of the string used to find the optix device
    //     programs that operate on this type */
    // std::string getTypeString() const { return "BSAMR"; };
    
    struct Block
    {
      uint32_t ID{UINT_MAX};
      box3i bounds;
      int level;
      int scalarOffset;
      range1f valueRange;

      inline __both__
      float getScalar(const float *scalarBuffer, int ix, int iy, int iz) const
      {
        const vec3i blockSize = this->numCells();
        const int idx
          = scalarOffset
          + ix
          + iy * blockSize.x
          + iz * blockSize.x*blockSize.y;
        return scalarBuffer[idx];
      }

      inline __both__
      int cellSize() const
      { return 1<<level; }

      inline __both__
      vec3i numCells() const
      { return bounds.upper-bounds.lower+vec3i(1); }

      inline __both__
      box4f worldBounds() const
      {
        box4f wb;
        wb.lower = vec4f(vec3f(bounds.lower*cellSize()),valueRange.lower);
        wb.upper = vec4f(vec3f((bounds.upper+1)*cellSize()),valueRange.upper);
        return wb;
      }

      inline __both__
      box4f filterDomain() const
      {
        const vec3f cellSize2(cellSize()*0.5f);
        box4f fd;
        fd.lower = vec4f(vec3f(bounds.lower*cellSize())-cellSize2,valueRange.lower);
        fd.upper = vec4f(vec3f((bounds.upper+1)*cellSize())+cellSize2,valueRange.upper);
        return fd;
      }

      inline __both__
      box3f cellBounds(const vec3i cellID) const
      {
        box3f cb;
        cb.lower = vec3f((bounds.lower+cellID)*cellSize());
        cb.upper = vec3f((bounds.lower+cellID+1)*cellSize());
        return cb;
      }
    };

    struct DD : public ScalarField::DD {

      static void addVars(std::vector<OWLVarDecl> &vars, int base);
      
      /* compute basis function contribution of given block at point P, and add
         that to 'sumWeightedValues' and 'sumWeights'. returns true if P is
         inside the block *filter domain*, false if outside (in which case the
         out params are not defined) */
      inline __both__ bool addBasisFunctions(float &sumWeightedValues,
                                             float &sumWeights,
                                             uint32_t bid,
                                             vec3f P) const;

      /* assemble block from SoA arrays, for the given linear idx */
      inline __both__ Block getBlock(int index) const;

      const box3i    *blockBounds;
      const int      *blockLevels;
      const int      *blockOffsets;
      const float    *blockScalars;
      const uint32_t *blockIDs;
      const range1f  *valueRanges;
      int             numBlocks;
    };

    // std::vector<OWLVarDecl> getVarDecls(uint32_t myOfs) override;
    void setVariables(OWLGeom geom) override;

    void buildMCs(MCGrid &macroCells) override;
    
    /*! computes, on specified device, the basis filter domains and - if
      d_primRanges is non-null - the primitmives ranges. d_primBounds
      and d_primRanges (if non-null) must be pre-allocated and
      writeaable on specified device */
    void computeBlockFilterDomains(int deviceID,
                                   box3f *d_primBounds,
                                   range1f *d_primRanges=0);

    BlockStructuredField(DataGroup *owner,
                         DevGroup *devGroup,
                         std::vector<box3i> &blockBounds,
                         std::vector<int> &blockLevels,
                         std::vector<int> &blockOffsets,
                         std::vector<float> &blockScalars);

    DD getDD(int devID);

    VolumeAccel::SP createAccel(Volume *volume) override;

    std::vector<box3i>    blockBounds;
    std::vector<int>      blockLevels;
    std::vector<int>      blockOffsets;
    std::vector<float>    blockScalars;
    std::vector<uint32_t> blockIDs;
    std::vector<range1f>  valueRanges;

    OWLBuffer blockBoundsBuffer  = 0;
    OWLBuffer blockLevelsBuffer  = 0;
    OWLBuffer blockOffsetsBuffer = 0;
    OWLBuffer blockScalarsBuffer = 0;
    OWLBuffer blockIDsBuffer     = 0;
    OWLBuffer valueRangesBuffer  = 0;
  };

  /* compute basis function contribution of given block at point P, and add
     that to 'sumWeightedValues' and 'sumWeights'. returns true if P is inside
     the block *filter domain*, false if outside (in which case the out params
     are not defined) */
  inline __both__
  bool BlockStructuredField::DD::addBasisFunctions(float &sumWeightedValues,
                                                   float &sumWeights,
                                                   uint32_t bid,
                                                   vec3f P) const
  {
    const auto &block = getBlock(bid);
    const box3f brickBounds = getBox(block.worldBounds());
    const box3f brickDomain = getBox(block.filterDomain());
    const vec3i blockSize = block.numCells();

    if (brickDomain.contains(P)) {
      const vec3f localPos = (P-brickBounds.lower) / vec3f((float)block.cellSize()) - 0.5f;
      vec3i idx_lo   = vec3i((int)floorf(localPos.x), (int)floorf(localPos.y), (int)floorf(localPos.z));
      idx_lo = max(vec3i(-1), idx_lo);
      const vec3i idx_hi   = idx_lo + vec3i(1);
      const vec3f frac     = localPos - vec3f(idx_lo);
      const vec3f neg_frac = vec3f(1.f) - frac;

      // #define INV_CELL_WIDTH invCellWidth
      #define INV_CELL_WIDTH 1.f
      if (idx_lo.z >= 0 && idx_lo.z < blockSize.z) {
        if (idx_lo.y >= 0 && idx_lo.y < blockSize.y) {
          if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_lo.x,idx_lo.y,idx_lo.z);
            const float weight = (neg_frac.z)*(neg_frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_hi.x,idx_lo.y,idx_lo.z);
            const float weight = (neg_frac.z)*(neg_frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
        if (idx_hi.y < blockSize.y) {
          if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_lo.x,idx_hi.y,idx_lo.z);
            const float weight = (neg_frac.z)*(frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_hi.x,idx_hi.y,idx_lo.z);
            const float weight = (neg_frac.z)*(frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
      }
        
      if (idx_hi.z < blockSize.z) {
        if (idx_lo.y >= 0 && idx_lo.y < blockSize.y) {
          if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_lo.x,idx_lo.y,idx_hi.z);
            const float weight = (frac.z)*(neg_frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_hi.x,idx_lo.y,idx_hi.z);
            const float weight = (frac.z)*(neg_frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
        if (idx_hi.y < blockSize.y) {
          if (idx_lo.x >= 0 && idx_lo.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_lo.x,idx_hi.y,idx_hi.z);
            const float weight = (frac.z)*(frac.y)*(neg_frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
          if (idx_hi.x < blockSize.x) {
            const float scalar = block.getScalar(blockScalars,idx_hi.x,idx_hi.y,idx_hi.z);
            const float weight = (frac.z)*(frac.y)*(frac.x);
            sumWeights += weight;
            sumWeightedValues += weight*scalar;
          }
        }
      }
      return true;
    }
    return false;
  }

  inline __both__
  BlockStructuredField::Block BlockStructuredField::DD::getBlock(int index) const
  {
    Block block;
   
    block.ID           = blockIDs[index];
    block.bounds       = blockBounds[index];
    block.level        = blockLevels[index];
    block.scalarOffset = blockOffsets[index];
    block.valueRange   = valueRanges[index];

    return block;
  }
}
