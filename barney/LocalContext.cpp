// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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
#include "barney/render/RayQueue.h"

#if defined(BARNEY_RTC_EMBREE) && defined(BARNEY_RTC_OPTIX)
# error "should not have both backends on at the same time!?"
#endif

namespace barney_api {
#if BARNEY_RTC_EMBREE
  extern "C" {
    Context *createContext_embree(const std::vector<int> &dgIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *embree (cpu)* context" << std::endl;
      std::vector<int> gpuIDs = { 0 };
      return new BARNEY_NS::LocalContext(dgIDs,gpuIDs);
    }
  }
#endif
#if BARNEY_RTC_OPTIX
  extern "C" {
    Context *createContext_optix(const std::vector<int> &dgIDs,
                                 int numGPUs, const int *_gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *optix* context" << std::endl;
      if (numGPUs == -1)
        BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
      std::vector<int> gpuIDs;
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back(_gpuIDs?_gpuIDs[i]:i);
      Context *ctx = new BARNEY_NS::LocalContext(dgIDs,gpuIDs);
      return ctx;
    }
  } 
#endif
#if BARNEY_RTC_CUDA
  extern "C" {
    Context *createContext_cuda(const std::vector<int> &dgIDs,
                                 int numGPUs, const int *_gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *(native-)cuda* context" << std::endl;
      if (numGPUs == -1)
        BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
      std::vector<int> gpuIDs;
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back(_gpuIDs?_gpuIDs[i]:i);
      Context *ctx = new BARNEY_NS::LocalContext(dgIDs,gpuIDs);
      return ctx;
    }
  } 
#endif
}

namespace BARNEY_NS {

  LocalContext::LocalContext(const std::vector<int> &dataGroupIDs,
                             const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs,0,1)
  {
    for (int i=0;i<devices->size();i++) {
      (*devices)[i]->globalRank = i;
      (*devices)[i]->globalSize = devices->size();
    }
  }

  LocalContext::~LocalContext()
  {
    /* not doing anything, but leave this in to ensure that derived
       classes' destrcutors get called !*/
  }

  std::shared_ptr<barney_api::FrameBuffer> LocalContext::createFrameBuffer(int owningRank)
  {
    assert(owningRank == 0);
    return std::make_shared<LocalFB>(this,devices);
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int LocalContext::numRaysActiveGlobally()
  {
    return numRaysActiveLocally();
  }

  bool LocalContext::forwardRays(bool needHitIDs)
  {
    const int numSlots = (int)perSlot.size();
    if (numSlots == 1) {
      // do NOT copy or swap. rays are in trace queue, which is also
      // the shade read queue, so nothing to do.
      //
      // no more trace rounds required: return false
      return false;
    }
    
    const int numDevices = (int)devices->size();
    const int dgSize = numDevices / numSlots;
    std::vector<int> numCopied(numDevices);
    for (auto device : *devices) {
      int devID = device->contextRank;
      SetActiveGPU forDuration(device);

      int nextID = (devID + dgSize) % numDevices;
      auto nextDev = (*devices)[nextID];

      int count = device->rayQueue->numActive;
      numCopied[nextID] = count;
      auto &src = device->rayQueue->traceAndShadeReadQueue;
      auto &dst = nextDev->rayQueue->receiveAndShadeWriteQueue;
      std::cout << "#### COPYING RAYS " << src.rays << " -> " << dst.rays << " #=" << count << std::endl;
      device->rtc->copyAsync(dst.rays,src.rays,count*sizeof(Ray));
      if (needHitIDs)
        device->rtc->copyAsync(dst.hitIDs,src.hitIDs,count*sizeof(*dst.hitIDs));
    }

    for (auto device : *devices) {
      int devID = device->contextRank;
      device->sync();
      device->rayQueue->swapAfterCycle(numTimesForwarded % numSlots, numSlots);
      device->rayQueue->numActive = numCopied[devID];
    }
    
    ++numTimesForwarded;
    return (numTimesForwarded % numSlots) != 0;
  }

  void LocalContext::render(Renderer    *renderer,
                            GlobalModel *model,
                            Camera      *camera,
                            FrameBuffer *fb)
  {
    assert(model);
    assert(fb);

    // render all tiles, in tile format and writing into accum buffer
    renderTiles(renderer,model,camera,fb);

    // convert all tiles from accum to RGBA
    finalizeTiles(fb);
    
    // ------------------------------------------------------------------
    // done rendering, let the frame buffer know about it, so it can
    // do whatever needs doing with the latest finalized tiles
    // ------------------------------------------------------------------
    fb->finalizeFrame();
  }

}
