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

#include "barney/fb/LocalFB.h"

namespace barney {
  LocalFB::LocalFB(Context *context)
    : FrameBuffer(context, true)
  {
  }

  void LocalFB::resize(vec2i size,
                       uint32_t channels)
  {
    FrameBuffer::resize(size,channels);
    if (gatheredTilesOnOwner.compressedTiles)
      BARNEY_CUDA_CALL(Free(gatheredTilesOnOwner.compressedTiles));
    if (gatheredTilesOnOwner.tileDescs)
      BARNEY_CUDA_CALL(Free(gatheredTilesOnOwner.tileDescs));

    // do NOT set active device - it's whatever the app used!
    // SetActiveDevice forDuration(perDev[0]->device);
    int sumTiles = 0;
    for (auto dev : perDev)
      sumTiles += dev->numActiveTiles;

    gatheredTilesOnOwner.numActiveTiles = sumTiles;
    BARNEY_CUDA_CALL(Malloc(&gatheredTilesOnOwner.compressedTiles,
                            sumTiles*sizeof(*gatheredTilesOnOwner.compressedTiles)));
    BARNEY_CUDA_CALL(Malloc(&gatheredTilesOnOwner.tileDescs,
                            sumTiles*sizeof(*gatheredTilesOnOwner.tileDescs)));
    sumTiles = 0;
    for (auto dev : perDev) {
      BARNEY_CUDA_CALL(Memcpy(gatheredTilesOnOwner.tileDescs+sumTiles,
                              dev->tileDescs,
                              dev->numActiveTiles*sizeof(*gatheredTilesOnOwner.tileDescs),
                              cudaMemcpyDefault));
      sumTiles += dev->numActiveTiles;
    }
  }
  
  void LocalFB::ownerGatherCompressedTiles()
  {
    // do NOT set active device - it's whatever the app used!
    // SetActiveDevice forDuration(perDev[0]->device);
    int sumTiles = 0;
    for (auto dev : perDev) {
      BARNEY_CUDA_CALL(Memcpy(gatheredTilesOnOwner.compressedTiles+sumTiles,
                              dev->compressedTiles,
                              dev->numActiveTiles*sizeof(*gatheredTilesOnOwner.compressedTiles),
                              cudaMemcpyDefault));
      sumTiles += dev->numActiveTiles;
    }
    gatheredTilesOnOwner.numActiveTiles = sumTiles;    
  }
  
  LocalFB::~LocalFB()
  {
    BARNEY_CUDA_CALL_NOTHROW(Free(gatheredTilesOnOwner.compressedTiles));
    BARNEY_CUDA_CALL_NOTHROW(Free(gatheredTilesOnOwner.tileDescs));
  }
  
}
