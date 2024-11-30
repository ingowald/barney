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

#include "barney/fb/TiledFB.h"
#include "barney/fb/FrameBuffer.h"
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>

// #include "optix_host.h"
// #include "optix_stubs.h"

namespace barney {

  TiledFB::SP TiledFB::create(Device::SP device, FrameBuffer *owner)
  {
    return std::make_shared<TiledFB>(device, owner);
  }

  TiledFB::TiledFB(Device::SP device, FrameBuffer *owner)
    : device(device),
      owner(owner)
  {}

  TiledFB::~TiledFB()
  { free(); }

  void TiledFB::free()
  {
    SetActiveGPU forDuration(device);
    if (accumTiles)  {
      BARNEY_CUDA_CALL(Free(accumTiles));
      accumTiles = nullptr;
    }
    if (compressedTiles) {
      BARNEY_CUDA_CALL(Free(compressedTiles));
      compressedTiles = nullptr;
    }
    if (tileDescs) {
      BARNEY_CUDA_CALL(Free(tileDescs));
      tileDescs = nullptr;
    }
  }

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
    free();
    SetActiveGPU forDuration(device);

    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTiles
      = device
      ? divRoundUp(numTiles.x*numTiles.y - device->globalIndex,
                   device->globalIndexStep)
      : 0;
#if 0
    BARNEY_CUDA_CALL(MallocManaged(&accumTiles, numActiveTiles * sizeof(AccumTile)));
    BARNEY_CUDA_CALL(MallocManaged(&compressedTiles, numActiveTiles * sizeof(CompressedTile)));
    BARNEY_CUDA_CALL(MallocManaged(&tileDescs,  numActiveTiles * sizeof(TileDesc)));
#else
    BARNEY_CUDA_CALL(Malloc(&accumTiles, numActiveTiles * sizeof(AccumTile)));
    BARNEY_CUDA_CALL(Malloc(&compressedTiles, numActiveTiles * sizeof(CompressedTile)));
    BARNEY_CUDA_CALL(Malloc(&tileDescs,  numActiveTiles * sizeof(TileDesc)));
#endif

    BARNEY_CUDA_SYNC_CHECK();
    if (numActiveTiles)
      CHECK_CUDA_LAUNCH(setTileCoords,
                        divRoundUp(numActiveTiles,1024),1024,0,device?device->launchStream:0,
                        //
                        tileDescs,numActiveTiles,numTiles,
                        device->globalIndex,device->globalIndexStep);
      // setTileCoords<<<divRoundUp(numActiveTiles,1024),1024,0,
      // device?device->launchStream:0>>>
      //   (tileDescs,numActiveTiles,numTiles,
      //    device->globalIndex,device->globalIndexStep);
    BARNEY_CUDA_SYNC_CHECK();
  }

  // ==================================================================

  __global__ void g_compressTiles(CompressedTile *compressedTiles,
                                  AccumTile *accumTiles,
                                  float      accumScale)
  {
    int pixelID = threadIdx.x;
    int tileID  = blockIdx.x;

    vec4f color = vec4f(accumTiles[tileID].accum[pixelID])*accumScale;
    float scale = reduce_max(color);
    color *= 1./scale;
    compressedTiles[tileID].scale[pixelID] = scale;
    compressedTiles[tileID].normal[pixelID].set(accumTiles[tileID].normal[pixelID]);

    uint32_t rgba32
      = owl::make_rgba(color);

    compressedTiles[tileID].rgba[pixelID] = rgba32;
    compressedTiles[tileID].depth[pixelID] = accumTiles[tileID].depth[pixelID];
  }

  /*! write this tiledFB's tiles into given "compressed" frame buffer
    (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
    pixels) */
  void TiledFB::finalizeTiles()
  {
    SetActiveGPU forDuration(device);
    if (numActiveTiles > 0)
      CHECK_CUDA_LAUNCH(g_compressTiles,
                        numActiveTiles,pixelsPerTile,0,device->launchStream,
                        compressedTiles,accumTiles,1.f/(owner->accumID));
      // g_compressTiles<<<numActiveTiles,pixelsPerTile,0,device->launchStream>>>
      //   (compressedTiles,accumTiles,1.f/(owner->accumID));
    BARNEY_CUDA_SYNC_CHECK();
  }

}
