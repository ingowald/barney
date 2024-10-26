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

#include "barney/LocalContext.h"
#include "barney/fb/LocalFB.h"

namespace barney {

  LocalContext::LocalContext(const std::vector<int> &dataGroupIDs,
                             const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs,0,1)
  {}

  LocalContext::~LocalContext()
  { /* not doing anything, but leave this in to ensure that derived
       classes' destrcutors get called !*/}

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
    const int numSlots = (int)perSlot.size();
    if (numSlots == 1) {
      // do NOT copy or swap. rays are in trace queue, which is also
      // the shade read queue, so nothing to do.
      //
      // no more trace rounds required: return false
      return false;
    }

    const int numDevices = (int)devices.size();
    const int dgSize = numDevices / numSlots;
    std::vector<int> numCopied(numDevices);
    for (int devID=0;devID<numDevices;devID++) {
      auto thisDev = devices[devID];
      SetActiveGPU forDuration(thisDev->device);

      int nextID = (devID + dgSize) % numDevices;
      auto nextDev = devices[nextID];

      int count = nextDev->rays.numActive;
      numCopied[devID] = count;
      Ray *src = nextDev->rays.traceAndShadeReadQueue;
      Ray *dst = thisDev->rays.receiveAndShadeWriteQueue;
      BARNEY_CUDA_CALL(MemcpyAsync(dst,src,count*sizeof(Ray),
                                   cudaMemcpyDefault,
                                   thisDev->device->launchStream));
    }

    for (auto dev : devices) dev->sync();

    for (int devID=0;devID<numDevices;devID++) {
      auto thisDev = devices[devID];
      thisDev->launch_sync();
      thisDev->rays.swap();
      thisDev->rays.numActive = numCopied[devID];
    }

    for (auto dev : devices) dev->sync();

    ++numTimesForwarded;
    return (numTimesForwarded % numSlots) != 0;
  }

  void LocalContext::render(GlobalModel *model,
                            const Camera::DD &camera,
                            FrameBuffer *fb,
                            int pathsPerPixel)
  {
    assert(model);
    assert(fb);

    // render all tiles, in tile format and writing into accum buffer
    renderTiles(model,camera,fb,pathsPerPixel);
    for (auto dev : devices) dev->sync();

    // convert all tiles from accum to RGBA
    finalizeTiles(fb);
    for (auto dev : devices) dev->sync();

    // ------------------------------------------------------------------
    // tell all GPUs to write their final pixels
    // ------------------------------------------------------------------
    // ***NO*** active device here
    LocalFB *localFB = (LocalFB*)fb;
    localFB->ownerGatherFinalTiles();
    TiledFB::writeFinalPixels(
# if DENOISE
#  if DENOISE_OIDN
                              fb->denoiserInput,
                              fb->denoiserAlpha,
#  else
                              fb->denoiserInput,
#  endif
# else
                              localFB->finalFB,
# endif
                              localFB->finalDepth,
# if DENOISE_NORMAL
                              fb->denoiserNormal,
# endif
                              localFB->numPixels,
                              localFB->rank0gather.finalTiles,
                              localFB->rank0gather.tileDescs,
                              localFB->rank0gather.numActiveTiles,
                              fb->showCrosshairs);
    // ------------------------------------------------------------------
    // wait for all GPUs to complete, so pixels are all written before
    // we return and/or copy to app
    // ------------------------------------------------------------------
    for (int localID = 0; localID < devices.size(); localID++)
      devices[localID]->launch_sync();


# if DENOISE
    fb->denoise();
#endif

    // ------------------------------------------------------------------
    // copy final frame buffer to app's frame buffer memory
    // ------------------------------------------------------------------
    if (fb->hostFB && fb->finalFB) 
      BARNEY_CUDA_CALL(Memcpy(fb->hostFB,fb->finalFB,
                            fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                            cudaMemcpyDefault));
    if (fb->hostDepth && fb->finalDepth)
      BARNEY_CUDA_CALL(Memcpy(fb->hostDepth,fb->finalDepth,
                            fb->numPixels.x*fb->numPixels.y*sizeof(float),
                            cudaMemcpyDefault));

    for (auto dev : devices) dev->sync();

  }

}
