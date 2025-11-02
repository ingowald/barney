// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/globalTrace/RQSLocal.h"
#include "barney/render/RayQueue.h"

namespace BARNEY_NS {

  RQSLocal::RQSLocal(Context *context)
    : RQSBase(context)
  {
    for (auto gd : context->topo->allDevices)
      assert(gd.worker == 0);
  }

  bool RQSLocal::forwardRays(bool needHitIDs)
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
