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
// #include <optix.h>
// #include <optix_function_table.h>
// #include <optix_stubs.h>

namespace barney {

  TiledFB::SP TiledFB::create(Device *device, FrameBuffer *owner)
  {
    return std::make_shared<TiledFB>(device, owner);
  }

  TiledFB::TiledFB(Device *device, FrameBuffer *owner)
    : device(device),
      owner(owner)
  {}

  TiledFB::~TiledFB()
  { free(); }

  void TiledFB::free()
  {
    SetActiveGPU forDuration(device);
    if (accumTiles)  {
      device->rtc->freeMem(accumTiles);
      accumTiles = nullptr;
    }
    if (compressedTiles) {
      device->rtc->freeMem(compressedTiles);
      compressedTiles = nullptr;
    }
    if (tileDescs) {
      device->rtc->freeMem(tileDescs);
      tileDescs = nullptr;
    }
  }

  struct SetTileCoords {
    /* kernel ARGS */
    TileDesc *tileDescs;
    int numActiveTiles;
    vec2i numTiles;
    int globalIndex;
    int globalIndexStep;

    /* kernel CODE */
    template<typename RTCore>
    inline __both__
    void run(const RTCore &rtCore)
    {
      int tid
        = rtCore.getThreadIdx().x
        + rtCore.getBlockIdx().x*rtCore.getBlockDim().x;
      if (tid >= numActiveTiles)
        return;
        
      int tileID = tid * globalIndexStep + globalIndex;
        
      int tile_x = tileID % numTiles.x;
      int tile_y = tileID / numTiles.x;
      tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
    }
  };

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
// #if 1
    auto rtc = device->rtc;
    accumTiles
      = (AccumTile *)rtc->alloc(numActiveTiles * sizeof(AccumTile));
    compressedTiles
      = (CompressedTile *)rtc->alloc(numActiveTiles * sizeof(CompressedTile));
    tileDescs
      = (TileDesc *)rtc->alloc(numActiveTiles * sizeof(TileDesc));
    SetTileCoords args = {
      tileDescs,
      numActiveTiles,
      numTiles,
      device->globalIndex,
      device->globalIndexStep
    };
    device->setTileCoords
      ->launch(divRoundUp(numActiveTiles,1024),1024,
               &args);
  }

  // ==================================================================


  struct CompressTiles {
    CompressedTile *compressedTiles;
    AccumTile      *accumTiles;
    float           accumScale;

    template<typename RTCore>
    inline __both__
    void run(const RTCore &rtCore)
    {
      int pixelID = rtCore.getThreadIdx().x;
      int tileID  = rtCore.getBlockIdx().x;

      vec4f color = vec4f(accumTiles[tileID].accum[pixelID])*accumScale;
      vec4f org = color;
      float scale = reduce_max(color);
      color *= 1.f/scale;
      compressedTiles[tileID].scale[pixelID] = scale;
      compressedTiles[tileID].normal[pixelID]
        .set(accumTiles[tileID].normal[pixelID]);

      
      uint32_t rgba32
        = make_rgba(color);

      compressedTiles[tileID].rgba[pixelID]
        = rgba32;
      compressedTiles[tileID].depth[pixelID]
        = accumTiles[tileID].depth[pixelID];
    }
  };

  /*! write this tiledFB's tiles into given "compressed" frame buffer
    (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
    pixels) */
  void TiledFB::finalizeTiles_launch()
  {
    SetActiveGPU forDuration(device);
    if (numActiveTiles > 0) {
      CompressTiles args = {
        compressedTiles,
        accumTiles,
        1.f/(owner->accumID)
      };
      device->compressTiles
        ->launch(numActiveTiles,pixelsPerTile,
                 &args);       
    }
  }
  
}

RTC_DECLARE_COMPUTE(setTileCoords,barney::SetTileCoords);
RTC_DECLARE_COMPUTE(compressTiles,barney::CompressTiles);


  
