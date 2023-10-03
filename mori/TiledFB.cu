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

#include "mori/TiledFB.h"

namespace mori {

  TiledFB::SP TiledFB::create(int gpuID,
                              int tileIndexOffset,
                              int tileIndexScale)
  {
    return std::make_shared<TiledFB>(gpuID,
                                     tileIndexOffset,
                                     tileIndexScale);
  }
  
  TiledFB::TiledFB(int gpuID,
                   int tileIndexOffset,
                   int tileIndexScale)
    : gpuID(gpuID),
      tileIndexOffset(tileIndexOffset),
      tileIndexScale(tileIndexScale)
  {}

  void TiledFB::resize(vec2i newSize)
  {
    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTiles
      = (numTiles.x*numTiles.y-tileIndexOffset)
      / tileIndexScale;
    if (tiles) 
      MORI_CUDA_CALL(Free(tiles));
    MORI_CUDA_CALL(MallocManaged(&tiles, numActiveTiles * sizeof(Tile)));
  }
  

    /*! write this tiledFB's tiles into given "final" frame buffer
        (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
        pixels) */
  void TiledFB::writeFinal(uint32_t *finalFB, cudaStream_t stream)
  {
    PING;
  }
}
