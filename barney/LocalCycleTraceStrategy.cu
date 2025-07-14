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

#include "barney/LocalCycleTraceStrategy.h"
#include "barney/render/RayQueue.h"

namespace BARNEY_NS {

  LocalCycleTraceStrategy::LocalCycleTraceStrategy(Context *context)
    : RayQueueCycleTraceStrategy(context)
  {
    assert(!context->globalDeviceInfos.empty());
    for (auto gd : context->topo->allDevices)
      assert(gd.worker == 0);
  }
  
  void RayQueueCycleTraceStrategy::traceRays(GlobalModel *model,
                                             uint32_t rngSeed,
                                             bool needHitIDs)
  {
    while (true) {
      if (FromEnv::get()->logQueues) 
        std::cout << "----- glob-trace -> locally) "
                  << " -----------" << std::endl;
      context->traceRaysLocally(model, rngSeed, needHitIDs);
      const bool needMoreTracing = forwardRays(needHitIDs);
      if (needMoreTracing)
        continue;
      break;
    }
  }


  bool RayQueueCycleTraceStrategy::forwardRays(bool needHitIDs)
  {
    const int numSlots = (int)context->perSlot.size();
    if (numSlots == 1) {
      // do NOT copy or swap. rays are in trace queue, which is also
      // the shade read queue, so nothing to do.
      //
      // no more trace rounds required: return false
      return false;
    }
    
    const int numDevices = (int)context->devices->size();
    // const int dgSize = numDevices / numSlots;
    // const int dgSize = 
    std::vector<int> numCopied(numDevices);
    for (auto device : *context->devices) {
      const PLD &pld = *getPLD(device);
      // int devID = device->contextRank();
      SetActiveGPU forDuration(device);
      // auto &rqs = device->rqs;
      int nextID = pld.recvPartner->local;//rqs.recvWorkerLocal;
      auto nextDev = (*context->devices)[nextID];

      int count = device->rayQueue->numActive;
      numCopied[nextID] = count;
      auto &src = device->rayQueue->traceAndShadeReadQueue;
      auto &dst = nextDev->rayQueue->receiveAndShadeWriteQueue;
      device->rtc->copyAsync(dst.rays,src.rays,count*sizeof(Ray));
      if (needHitIDs)
        device->rtc->copyAsync(dst.hitIDs,src.hitIDs,count*sizeof(*dst.hitIDs));
    }

    for (auto device : *context->devices) {
      int devID = device->contextRank();
      device->sync();
      device->rayQueue->swapAfterCycle(numTimesForwarded % numSlots, numSlots);
      device->rayQueue->numActive = numCopied[devID];
    }
    
    ++numTimesForwarded;
    return (numTimesForwarded % numSlots) != 0;
  }

}
