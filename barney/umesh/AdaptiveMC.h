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

#include <array>

#include "barney/umesh/common/UMeshField.h"
#include "barney/Object.h"
#include "barney/volume/TransferFunction.h"
#include "barney/volume/ScalarField.h"

namespace barney {

  inline __both__ vec3i operator<<(vec3i v, int s)
  { return { v.x<<s, v.y<<s, v.z<<s }; }

  struct DataGroup;
  struct Volume;
  struct ScalarField;
  struct MCGrid;
  
  typedef std::array<int,4> TetIndices;
  typedef std::array<int,5> PyrIndices;
  typedef std::array<int,6> WedIndices;
  typedef std::array<int,8> HexIndices;

  struct AdaptiveMC {
    struct Leaf {
      vec3i cellID;
      int   level;
      union {
        struct {
          inline __device__ void clear() { numElements = 0; sumEdgeLengths = 0; }
          float sumEdgeLengths;
          int   numElements;
        } duringBuild;
        struct {
          range1f valueRange;
        } onceBuilt;
      };
    };
        
    struct Node {
      vec3i cellID;
      int   level;
      uint32_t isLeaf :  1;
      /*! pinto tinto leaf list, or into nodes list */
      uint32_t offset  :31;
    };
    struct NextSplitJob {
      uint32_t nodeID;
      float    priority;
    };
    struct Forest {
      Forest() {};
      box3f     worldBounds;
      vec3f     rootCellWidth;
      vec3i     rootDims;
      // Cell     *roots = 0;
      Node     *nodes = 0;
      Leaf     *leaves = 0;
      float    *majorants = 0;
      uint32_t numNodes  = 0;
      uint32_t numLeaves = 0;

      inline __both__ box3f getCellBounds(vec3i cellID, int level) const
      {
        vec3i intCoords = cellID;// << leaf.level;
        box3f cellBounds;
        vec3f cellWidth = 1.f/(1<<level);
        // RELATIVE
        cellBounds.lower = vec3f(intCoords) * cellWidth;
        cellBounds.upper = cellBounds.lower + cellWidth;

        vec3f blockWidth = rootCellWidth;
        // vec3f blockWidth = forest->worldBounds.size()*rcp(vec3f(forest->rootDims));
        cellBounds.lower = worldBounds.lower + blockWidth * cellBounds.lower;
        cellBounds.upper = worldBounds.lower + blockWidth * cellBounds.upper;
        return cellBounds;
      }
      
      
      inline __both__ box3f getLeafBounds(Leaf leaf) const
      {
        return getCellBounds(leaf.cellID,leaf.level);
      }
    };

    static void build(UMeshField *mesh);
             
  };
  
}
