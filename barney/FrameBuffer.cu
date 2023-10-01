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

#include "barney/FrameBuffer.h"

namespace barney {

  void FrameBuffer::resize(vec2i size)
  {
    fbSize = size;
    numTiles = divRoundUp(size,vec2i(tileSize));
    numActiveTiles
      = (numTiles.x*numTiles.y-tileIndexOffset)
      / tileIndexScale;
    if (tiles) 
      MORI_CUDA_CALL(Free(tiles));
    MORI_CUDA_CALL(MallocManaged(&tiles, numActiveTiles * sizeof(Tile)));

    if (finalFB)
      MORI_CUDA_CALL(Free(finalFB));
    MORI_CUDA_CALL(MallocManaged(&finalFB, fbSize.x*fbSize.y * sizeof(uint32_t)));
    
    PING;
    PRINT(numActiveTiles);
    PRINT(tiles);
  }
  
}
