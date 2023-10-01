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

#include "barney/Model.h"
#include "barney/LocalFB.h"
#include "owl/owl.h"

namespace barney {

  
  __global__ void render_testFrame(vec2i fbSize,
                                   FrameBuffer::Tile *tiles,
                                   vec2i numTiles,
                                   int tileIndexOffset,
                                   int tileIndexScale,
                                   uint32_t *appFB)
  {
    
    int localTileIdx = blockIdx.x;
    int globalTileIdx = tileIndexScale * localTileIdx + tileIndexOffset;
    int tile_y = globalTileIdx / numTiles.x;
    int tile_x = globalTileIdx - tile_y * numTiles.x;
    int ix = threadIdx.x + tile_x * FrameBuffer::tileSize;
    int iy = threadIdx.y + tile_y * FrameBuffer::tileSize;

    bool dbg = (ix == 118 && iy == 123);
    if (dbg)
      printf("(%i %i) fb %i %i tile %i %i\n",
             ix,iy,fbSize.x,fbSize.y,tile_x,tile_y);
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;
    FrameBuffer::Tile &tile = tiles[localTileIdx];

    // int sx = 13*17;
    // int sy = 11*19;
    // float r = (ix % sx)/(sx-1.f);
    // float g = (iy % sy)/(sy-1.f);
    float r = ix / (fbSize.x-1.f);
    float g = iy / (fbSize.y-1.f);
    float b = 1.f - (ix+iy)/(fbSize.x+fbSize.y-1.f);

    tile.accum[threadIdx.y*FrameBuffer::tileSize+threadIdx.x] = make_float4(r,g,b,1.f);

    appFB[ix + iy * fbSize.x] = owl::make_rgba(vec3f(r,g,b));
  }
  
  void Model::render(const BNCamera *camera,
                     FrameBuffer *fb,
                     uint32_t *appFB)
  {
    assert(fb);
    assert(camera);
    assert(appFB);
    PRINT(fb->numTiles);
    assert(fb->tiles);
    PRINT(appFB);
    PRINT(fb->numActiveTiles);
    render_testFrame
      <<<fb->numActiveTiles,vec2i(fb->tileSize,fb->tileSize)>>>
      (fb->fbSize,fb->tiles,fb->numTiles,0,1,fb->finalFB);
    
    MORI_CUDA_SYNC_CHECK();
    MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
                          fb->fbSize.x*fb->fbSize.y*sizeof(uint32_t),
                          cudaMemcpyDefault));
    MORI_CUDA_SYNC_CHECK();
  }

}
