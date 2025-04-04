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

#include "barney/fb/FrameBuffer.h"
#include "barney/common/MPIWrappers.h"

namespace BARNEY_NS {

  struct MPIContext;
  
  struct DistFB : public FrameBuffer {
    typedef std::shared_ptr<DistFB> SP;

    DistFB(MPIContext *context,
           const DevGroup::SP &devices,
           int owningRank);
    virtual ~DistFB() = default;
    
    void resize(vec2i size, uint32_t channels) override;

    void ownerGatherCompressedTiles() override;
    
    struct {
      std::vector<int> numTilesOnGPU;
      std::vector<int> firstTileOnGPU;
      int numGPUs;
    } ownerGather;
    // (world)rank that owns this frame buffer
    const int owningRank;
    const bool isOwner;
    const bool ownerIsWorker;
    MPIContext *context;
  };

}
