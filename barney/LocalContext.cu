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
  {}
  
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
    const int numDataGroups = perDG.size();
    if (numDataGroups == 1)
      return false;
    
    const int numDevices = devices.size();
    const int dgSize = numDevices / numDataGroups;
    int numCopied[numDevices];
    for (int devID=0;devID<numDevices;devID++) {
      auto thisDev = devices[devID];
      SetActiveGPU forDuration(thisDev->device);
      
      int nextID = (devID + dgSize) % numDevices;
      auto nextDev = devices[nextID];
      
      int count = nextDev->rays.numActive;
      // std::cout << "forwarding " << count << " rays between " << devID << " and " << nextID << std::endl;
      numCopied[devID] = count;
      Ray *src = nextDev->rays.readQueue;
      Ray *dst = thisDev->rays.writeQueue;
      BARNEY_CUDA_CALL(MemcpyAsync(dst,src,count*sizeof(Ray),
                                   cudaMemcpyDefault,
                                   thisDev->device->launchStream));
    }

    for (auto dev : devices) dev->sync();

    for (int devID=0;devID<numDevices;devID++) {
      auto thisDev = devices[devID];
      PING; fflush(0);
      thisDev->launch_sync();
      thisDev->rays.swap();
      thisDev->rays.numActive = numCopied[devID];
    }

    for (auto dev : devices) dev->sync();

    ++numTimesForwarded;
    return (numTimesForwarded % numDataGroups) != 0;
    // return false;
  }

  void LocalContext::render(Model *model,
                            const Camera &camera,
                            FrameBuffer *fb)
  {
    assert(model);
    assert(fb);

    // render all tiles, in tile format and writing into accum buffer
    renderTiles(model,camera,fb);
    for (auto dev : devices) dev->sync();
    
    // convert all tiles from accum to RGBA
    finalizeTiles(fb);
    for (auto dev : devices) dev->sync();

    // ------------------------------------------------------------------
    // tell all GPUs to write their final pixels
    // ------------------------------------------------------------------
    for (int localID = 0; localID < devices.size(); localID++) {
      auto &devFB = *fb->perDev[localID];
      TiledFB::writeFinalPixels(nullptr,//devFB.device.get(),
                                fb->finalFB,
                                fb->numPixels,
                                devFB.finalTiles,
                                devFB.tileDescs,
                                devFB.numActiveTiles);
    }
    for (auto dev : devices) dev->sync();
    
    // ------------------------------------------------------------------
    // wait for all GPUs to complete, so pixels are all written before
    // we return and/or copy to app
    // ------------------------------------------------------------------
    for (int localID = 0; localID < devices.size(); localID++)
      devices[localID]->launch_sync();

    // ------------------------------------------------------------------
    // copy final frame buffer to app's frame buffer memory
    // ------------------------------------------------------------------
    if (fb->hostFB != fb->finalFB)
      BARNEY_CUDA_CALL(Memcpy(fb->hostFB,fb->finalFB,
                            fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                            cudaMemcpyDefault));
    for (auto dev : devices) dev->sync();
  }
  
}
