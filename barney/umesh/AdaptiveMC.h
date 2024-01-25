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
    struct NextSplitCell {
      union {
        struct {
          uint32_t cell;
          float    priority;
        };
        uint64_t bits;
      };
    };
    struct Forest {
      Forest() {};
      box3f     worldBounds;
      vec3i     rootDims;
      // Cell     *roots = 0;
      Node     *nodes = 0;
      Leaf     *leaves = 0;
      float    *majorants = 0;
      uint32_t numNodes  = 0;
      uint32_t numLeaves = 0;
    };

    static void build(UMeshField::SP mesh);
             
  };
  
}
