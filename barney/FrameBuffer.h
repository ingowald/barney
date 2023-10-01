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

#include "barney/Context.h"

namespace barney {

  struct FrameBuffer : public Object {
    
    enum { tileSize = 32 };

    struct Tile {
      float4 accum[tileSize*tileSize];
    };
    struct DD {
      Tile *tiles;
    };
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "<FrameBuffer(base)>"; }
    
    virtual void resize(vec2i size);
    
    /*! number of (valid) pixels */
    vec2i fbSize;
    
    /*! number of tiles to cover the entire frame buffer; some on the
      right/bottom may be partly filled */
    vec2i numTiles        = { 0, 0 };
    int   numActiveTiles  = 0;
    int   tileIndexOffset = 0;
    int   tileIndexScale  = 1;
    Tile *tiles = 0;
    uint32_t *finalFB     = 0;
  };
}
