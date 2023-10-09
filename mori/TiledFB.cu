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
#include "owl/owl.h"

namespace mori {

  TiledFB::SP TiledFB::create(DeviceContext *device)
  {
    return std::make_shared<TiledFB>(device);
  }
  
  TiledFB::TiledFB(DeviceContext *device)
    : device(device)
  {}

  __global__ void setTileCoords(TileDesc *tileDescs,
                                int numActiveTiles,
                                vec2i numTiles,
                                int tileIndexOffset,
                                int tileIndexScale)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numActiveTiles)
      return;
    
    int tileID = tid * tileIndexScale + tileIndexOffset;

    int tile_x = tileID % numTiles.x;
    int tile_y = tileID / numTiles.x;
    tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
  }
  
  void TiledFB::resize(vec2i newSize)
  {
    SetActiveGPU forDuration(device);
    if (accumTiles)  {
      MORI_CUDA_CALL(Free(accumTiles));
      // MORI_CUDA_CALL(FreeAsync(accumTiles,device->stream));
      accumTiles = nullptr;
    }
    if (finalTiles) {
      MORI_CUDA_CALL(Free(finalTiles));
      // MORI_CUDA_CALL(FreeAsync(finalTiles,device->stream));
      finalTiles = nullptr;
    }
    if (tileDescs) {
      MORI_CUDA_CALL(Free(tileDescs));
      // MORI_CUDA_CALL(FreeAsync(tileDescs,device->stream));
      tileDescs = nullptr;
    }
    
    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTiles
      = divRoundUp(numTiles.x*numTiles.y - device->tileIndexOffset,
                   device->tileIndexScale);
    MORI_CUDA_CALL(Malloc(&accumTiles, numActiveTiles * sizeof(AccumTile)));
    MORI_CUDA_CALL(Malloc(&finalTiles, numActiveTiles * sizeof(FinalTile)));
    MORI_CUDA_CALL(MallocManaged(&tileDescs, numActiveTiles * sizeof(TileDesc)));
    // MORI_CUDA_CALL(MallocAsync(&accumTiles, numActiveTiles * sizeof(AccumTile),
    //                            device->stream));
    // MORI_CUDA_CALL(MallocAsync(&finalTiles, numActiveTiles * sizeof(FinalTile),
    //                            device->stream));
    // MORI_CUDA_CALL(MallocAsync(&tileDescs, numActiveTiles * sizeof(TileDesc),
    //                            device->stream));
    MORI_CUDA_SYNC_CHECK();
    if (numActiveTiles)
      setTileCoords<<<divRoundUp(numActiveTiles,1024),1024,0,device->stream>>>
        (tileDescs,numActiveTiles,numTiles,
         device->tileIndexOffset,device->tileIndexScale);
    MORI_CUDA_SYNC_CHECK();
  }

  __global__ void g_finalizeTiles(FinalTile *finalTiles,
                                  AccumTile *accumTiles)
  {
    int pixelID = threadIdx.x+tileSize*threadIdx.y;
    int tileID = blockIdx.x;
    finalTiles[tileID].rgba[pixelID]
      = owl::make_rgba(vec4f(accumTiles[tileID].accum[pixelID]));
  }

  __global__ void g_writeFinalPixels(uint32_t  *finalFB,
                                     vec2i      numPixels,
                                     FinalTile *finalTiles,
                                     TileDesc  *tileDescs)
  {
    int tileID = blockIdx.x;
    int ix = threadIdx.x + tileDescs[tileID].lower.x;
    int iy = threadIdx.y + tileDescs[tileID].lower.y;
    if (ix >= numPixels.x) return;
    if (iy >= numPixels.y) return;
    finalFB[ix + numPixels.x*iy]
      = finalTiles[tileID].rgba[threadIdx.x+threadIdx.y*tileSize];
  }
                                 
  void TiledFB::writeFinalPixels(uint32_t  *finalFB,
                                 vec2i      numPixels,
                                 FinalTile *finalTiles,
                                 TileDesc  *tileDescs,
                                 int        numTiles,
                                 cudaStream_t stream)
  {
    g_writeFinalPixels
      <<<numTiles,vec2i(tileSize),0,stream>>>
      (finalFB,numPixels,
       finalTiles,tileDescs);
  }
  
  /*! write this tiledFB's tiles into given "final" frame buffer
    (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
    pixels) */
  void TiledFB::finalizeTiles()
  {
    SetActiveGPU forDuration(device);
    g_finalizeTiles<<<numActiveTiles,vec2i(tileSize),0,device->stream>>>
      (finalTiles,accumTiles);
  }
}
