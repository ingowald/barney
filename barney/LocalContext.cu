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

#include "barney/LocalContext.h"
#include "barney/LocalFB.h"

namespace barney {

  LocalContext::LocalContext(const std::vector<int> &dataGroupIDs,
                             const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs)
  {
  }
  
  FrameBuffer *LocalContext::createFB() 
  { return initReference(LocalFB::create(this)); }

  __global__ void g_renderTiles(mori::AccumTile *tiles,
                                mori::TileDesc  *tileDescs,
                                int numTiles,
                                vec2i fbSize)
  {
    int tileID = blockIdx.x;
    vec2i tileOffset = tileDescs[tileID].lower;
    int ix = threadIdx.x + tileOffset.x;
    int iy = threadIdx.y + tileOffset.y;
    
    // if (threadIdx.x == 0 && threadIdx.y == 0)
    //   printf("tile %i ofs %i %i\n",
    //          tileID,tile             globalTileIdx,
    //          tile_
    //          bool dbg = (ix == 118 && iy == 123);
    //          if (dbg)
    //            printf("(%i %i) fb %i %i tile %i %i\n",
    //          ix,iy,fbSize.x,fbSize.y,tile_x,tile_y);
    
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;
    mori::AccumTile &tile = tiles[tileID];

    // int sx = 13*17;
    // int sy = 11*19;
    // float r = (ix % sx)/(sx-1.f);
    // float g = (iy % sy)/(sy-1.f);
    float r = ix / (fbSize.x-1.f);
    float g = iy / (fbSize.y-1.f);
    float b = 1.f - (ix+iy)/(fbSize.x+fbSize.y-1.f);

    tile.accum[threadIdx.y*mori::tileSize+threadIdx.x] = make_float4(r,g,b,1.f);
  }
  
  void renderTiles(Context *context,
                   int localID,
                   Model *model,
                   FrameBuffer *fb,
                   const BNCamera *camera)
  {
    auto &devFB = *fb->perGPU[localID];
    auto device = devFB.device;
    
    SetActiveGPU forDuration(device->gpuID);
    PING;
    PRINT(devFB.numActiveTiles);
    g_renderTiles
      <<<devFB.numActiveTiles,vec2i(mori::tileSize),0,device->stream>>>
      (devFB.accumTiles,
       devFB.tileDescs,
       devFB.numActiveTiles,
       devFB.numPixels);
  }
  
  void LocalContext::render(Model *model,
                            const BNCamera *camera,
                            FrameBuffer *fb,
                            uint32_t *appFB)
  {
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      // computation of Tile::accum color
      renderTiles(this,localID,model,fb,camera);
    
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      // accum to rgba conversion:
      fb->perGPU[localID]->finalizeTiles();

    for (int localID = 0; localID < gpuIDs.size(); localID++) {
      auto &devFB = *fb->perGPU[localID];
      SetActiveGPU forDuration(devFB.device);
      mori::TiledFB::writeFinalPixels(fb->finalFB,
                                      fb->numPixels,
                                      devFB.finalTiles,
                                      devFB.tileDescs,
                                      devFB.numActiveTiles,
                                      devFB.device->stream);
    }
    
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      fb->perGPU[localID]->sync();


    // render_testFrame
    //   <<<fb->numActiveTiles,vec2i(fb->mori::tileSize,fb->mori::tileSize)>>>
    //   (fb->fbSize,fb->tiles,fb->numTiles,0,1,fb->finalFB);
      
    //   MORI_CUDA_SYNC_CHECK();
    //   MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
    //                         fb->fbSize.x*fb->fbSize.y*sizeof(uint32_t),
    //                         cudaMemcpyDefault));
    MORI_CUDA_SYNC_CHECK();
    PING;
    MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
                          fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                          cudaMemcpyDefault));
    MORI_CUDA_SYNC_CHECK();
  }
  
}
