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

#include "barney/DeviceGroup.h"

namespace barney {

#define FB_HAVE_DEPTH 1
  struct FrameBuffer;
  
  enum { tileSize = 32 };
  enum { pixelsPerTile = tileSize*tileSize };
  
  struct AccumTile {
    float4 accum[pixelsPerTile];
    float  depth[pixelsPerTile];
  };
  struct FinalTile {
    uint32_t rgba[pixelsPerTile];
    float    depth[pixelsPerTile];
  };
  struct TileDesc {
    vec2i lower;
  };
  
  struct TiledFB {
    typedef std::shared_ptr<TiledFB> SP;
    static SP create(Device::SP device, FrameBuffer *owner);

    TiledFB(Device::SP device, FrameBuffer *owner);
    
    void resize(vec2i newSize);

    /*! write this tiledFB's tiles into given "final" frame buffer
        (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
        pixels) */
    static
    void writeFinalPixels(Device    *device,
                          uint32_t  *finalFB,
                          float     *finalDepth,
                          vec2i      numPixels,
                          FinalTile *finalTiles,
                          TileDesc  *tileDescs,
                          int        numTiles);
    
    void finalizeTiles();

    /*! number of (valid) pixels */
    vec2i numPixels       = { 0,0 };
    
    /*! number of tiles to cover the entire frame buffer; some on the
      right/bottom may be partly filled */
    vec2i numTiles        = { 0, 0 };
    int   numActiveTiles  = 0;
    /*! lower-left pixel coordinate for given tile ... */
    TileDesc  *tileDescs  = 0;
    AccumTile *accumTiles = 0;
    FinalTile *finalTiles = 0;
    FrameBuffer *const owner;
    Device::SP const device;
  };
  
}
