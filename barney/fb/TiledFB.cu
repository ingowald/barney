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

#include "barney/fb/TiledFB.h"
#include "barney/fb/FrameBuffer.h"

namespace barney {

  TiledFB::SP TiledFB::create(Device::SP device, FrameBuffer *owner)
  {
    return std::make_shared<TiledFB>(device, owner);
  }
  
  TiledFB::TiledFB(Device::SP device, FrameBuffer *owner)
    : device(device), owner(owner)
  {}

  __global__ void setTileCoords(TileDesc *tileDescs,
                                int numActiveTiles,
                                vec2i numTiles,
                                int globalIndex,
                                int globalIndexStep)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= numActiveTiles)
      return;
    
    int tileID = tid * globalIndexStep + globalIndex;

    int tile_x = tileID % numTiles.x;
    int tile_y = tileID / numTiles.x;
    tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
  }
  
  void TiledFB::resize(vec2i newSize)
  {
    SetActiveGPU forDuration(device);
    if (accumTiles)  {
      BARNEY_CUDA_CALL(Free(accumTiles));
      accumTiles = nullptr;
    }
    if (finalTiles) {
      BARNEY_CUDA_CALL(Free(finalTiles));
      finalTiles = nullptr;
    }
    if (tileDescs) {
      BARNEY_CUDA_CALL(Free(tileDescs));
      tileDescs = nullptr;
    }
    
    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTiles
      = device
      ? divRoundUp(numTiles.x*numTiles.y - device->globalIndex,
                   device->globalIndexStep)
      : 0;
    BARNEY_CUDA_CALL(Malloc(&accumTiles, numActiveTiles * sizeof(AccumTile)));
    BARNEY_CUDA_CALL(Malloc(&finalTiles, numActiveTiles * sizeof(FinalTile)));
    BARNEY_CUDA_CALL(Malloc(&tileDescs, numActiveTiles * sizeof(TileDesc)));
    BARNEY_CUDA_SYNC_CHECK();
    if (numActiveTiles)
      setTileCoords<<<divRoundUp(numActiveTiles,1024),1024,0,
      device?device->launchStream:0>>>
        (tileDescs,numActiveTiles,numTiles,
         device->globalIndex,device->globalIndexStep);
    BARNEY_CUDA_SYNC_CHECK();
  }

  // ==================================================================

  __global__ void g_finalizeTiles(FinalTile *finalTiles,
                                  AccumTile *accumTiles,
                                  float      accumScale)
  {
    int pixelID = threadIdx.x;
    int tileID  = blockIdx.x;

    vec4f color = vec4f(accumTiles[tileID].accum[pixelID])*accumScale;

    // int ix = pixelID%tileSize;
    // int iy = pixelID/tileSize;
    // color.x = (ix % 256)/255.f;
    // color.y = (iy % 256)/255.f;
    // color.z = 1.f;
    uint32_t rgba32
      = owl::make_rgba(color);
    
    finalTiles[tileID].rgba[pixelID] = rgba32;
    finalTiles[tileID].depth[pixelID] = accumTiles[tileID].depth[pixelID];
  }

  /*! write this tiledFB's tiles into given "final" frame buffer
    (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
    pixels) */
  void TiledFB::finalizeTiles()
  {
    SetActiveGPU forDuration(device);
    if (numActiveTiles > 0)
      g_finalizeTiles<<<numActiveTiles,pixelsPerTile,0,device->launchStream>>>
        (finalTiles,accumTiles,1.f/(owner->accumID));
  }


  // ==================================================================

  __global__ void g_writeFinalPixels(uint32_t  *finalFB,
                                     float     *finalDepth,
                                     vec2i      numPixels,
                                     FinalTile *finalTiles,
                                     TileDesc  *tileDescs)
  {
    int tileID = blockIdx.x;
    int ix = threadIdx.x + tileDescs[tileID].lower.x;
    int iy = threadIdx.y + tileDescs[tileID].lower.y;
    if (ix < 0 || ix >= numPixels.x) return;
    if (iy < 0 || iy >= numPixels.y) return;

    uint32_t pixelValue
      = finalTiles[tileID].rgba[threadIdx.x + tileSize*threadIdx.y];
    pixelValue |= 0xff000000;

    uint32_t ofs = ix + numPixels.x*iy;
    finalFB[ofs] = pixelValue;
    
    if (finalDepth)
      finalDepth[ofs] = finalTiles[tileID].depth[threadIdx.x + tileSize*threadIdx.y];
  }
                                 
  void TiledFB::writeFinalPixels(Device    *device,
                                 uint32_t  *finalFB,
                                 float     *finalDepth,
                                 vec2i      numPixels,
                                 FinalTile *finalTiles,
                                 TileDesc  *tileDescs,
                                 int        numTiles)
  {
    if (finalFB == 0) throw std::runtime_error("invalid finalfb of null!");

    /*! do NOT set active GPU: app might run on a different GPU than
        what we think of as GPU 0, and taht may or may not be
        writeable by what we might think of a "first" gpu */
    // SetActiveGPU forDuration(device);

    if (numTiles > 0)
      g_writeFinalPixels
        <<<numTiles,vec2i(tileSize),0,
      device?device->launchStream:0>>>
        (finalFB,finalDepth,numPixels,
         finalTiles,tileDescs);
  }
  
}
