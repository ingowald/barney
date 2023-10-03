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
  { return initReference(LocalFB::create(this,0,1)); }

  __global__ void g_renderTiles(mori::Tile *tiles)
  {
  }
  
  void renderTiles(Context *context,
                   int localID,
                   Model *model,
                   FrameBuffer *fb,
                   const BNCamera *camera)
  {
    auto &devFB = *fb->perGPU[localID];
    cudaStream_t stream = 0;
    g_renderTiles
      <<<devFB.numActiveTiles,vec2i(mori::tileSize),0,stream>>>
      (devFB.tiles);
  }
  
  void LocalContext::render(Model *model,
                       const BNCamera *camera,
                       FrameBuffer *fb,
                       uint32_t *appFB)
  {
    LocalFB *local = (LocalFB *)fb;

    for (int localID = 0; localID < gpuIDs.size(); localID++)
      renderTiles(this,localID,model,fb,camera);
    
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      fb->perGPU[localID]->writeFinal(local->finalFB,perGPU[localID]->stream);

    for (int localID = 0; localID < gpuIDs.size(); localID++)
      MORI_CUDA_CALL(StreamSynchronize(perGPU[localID]->stream));
    
    // render_testFrame
    //   <<<fb->numActiveTiles,vec2i(fb->mori::tileSize,fb->mori::tileSize)>>>
    //   (fb->fbSize,fb->tiles,fb->numTiles,0,1,fb->finalFB);
      
    //   MORI_CUDA_SYNC_CHECK();
    //   MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
    //                         fb->fbSize.x*fb->fbSize.y*sizeof(uint32_t),
    //                         cudaMemcpyDefault));
    //   MORI_CUDA_SYNC_CHECK();
    MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
                          fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                          cudaMemcpyDefault));
    MORI_CUDA_SYNC_CHECK();
  }
  
}
