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

#include "barney/FrameBuffer.h"
#include "barney/MPIWrappers.h"

namespace barney {

  struct MPIContext;
  
  struct DistFB : public FrameBuffer {
    typedef std::shared_ptr<DistFB> SP;

    DistFB(MPIContext *context,
           int owningRank);
    
    static SP create(MPIContext *context, int owningRank)
    { return std::make_shared<DistFB>(context,owningRank); }
    
    void resize(vec2i size, uint32_t *hostFB) override;

    void ownerGatherFinalTiles();
    
    struct {
      /*! list of *all* ranks' tileOffset, gathered (only at master) */
      mori::FinalTile *finalTiles = 0;
      TileDesc        *tileDescs = 0;
      std::vector<int> numTilesOnGPU;
      std::vector<int> firstTileOnGPU;
      int numActiveTiles;
      int numGPUs;
    } ownerGather;
    // (world)rank that owns this frame buffer
    const int owningRank;
    const bool isOwner;
    const bool ownerIsWorker;
    MPIContext *context;
  };

}
