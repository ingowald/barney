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
    : Context(dataGroupIDs,gpuIDs,0,1)
  {
  }
  
  FrameBuffer *LocalContext::createFB(int owningRank) 
  {
    assert(owningRank == 0);
    return initReference(LocalFB::create(this));
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int LocalContext::numRaysActiveGlobally()
  {
    return numRaysActiveLocally();
  }

  bool LocalContext::forwardRays()
  {
    for (auto mori : moris)
      mori->rays.numActive = 0;
    return false;
  }

  void LocalContext::render(Model *model,
                            const mori::Camera *camera,
                            FrameBuffer *fb)
  {
    assert(camera);
    assert(model);
    assert(fb);

    // render all tiles, in tile format and writing into accum buffer
    renderTiles(model,*camera,fb);
    // convert all tiles from accum to RGBA
    finalizeTiles(fb);

    // ------------------------------------------------------------------
    // tell all GPUs to write their final pixels
    // ------------------------------------------------------------------
    for (int localID = 0; localID < moris.size(); localID++) {
      auto &devFB = *fb->moris[localID];
      mori::TiledFB::writeFinalPixels(devFB.device.get(),
                                      fb->finalFB,
                                      fb->numPixels,
                                      devFB.finalTiles,
                                      devFB.tileDescs,
                                      devFB.numActiveTiles);
    }
    
    // ------------------------------------------------------------------
    // wait for all GPUs to complete, so pixels are all written before
    // we return and/or copy to app
    // ------------------------------------------------------------------
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      moris[localID]->launch_sync();

    // ------------------------------------------------------------------
    // copy final frame buffer to app's frame buffer memory
    // ------------------------------------------------------------------
    if (fb->hostFB != fb->finalFB)
      MORI_CUDA_CALL(Memcpy(fb->hostFB,fb->finalFB,
                            fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                            cudaMemcpyDefault));
    MORI_CUDA_SYNC_CHECK();
  }
  
}
