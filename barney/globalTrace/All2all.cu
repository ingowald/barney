// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "barney/globalTrace/All2all.h"
#include "barney/DeviceGroup.h"
#include "barney/render/RayQueue.h"

namespace BARNEY_NS {

  MPIAll2all::MPIAll2all(Context *context)
    : GlobalTraceImpl(context)
  {
    perLogical.resize(context->devices->numLogical);
  }
  
  MPIAll2all::PLD *MPIAll2all::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  void MPIAll2all::ensureAllOurQueuesAreLargeEnough()
  {
    for (auto device : *context->devices) {
      auto rtc = device->rtc;
      PRINT(device->rayQueue->size);
      PLD *pld = getPLD(device);
      size_t ourRequiredQueueSize
        = device->rayQueue->size * context->topo->islandSize();
      std::cout << "need to resize ray queues from " << pld->currentSize << " to " << ourRequiredQueueSize << std::endl;
      if (ourRequiredQueueSize > pld->currentSize) {
        if (pld->send.raysOnly) rtc->freeMem(pld->send.raysOnly);
        if (pld->recv.raysOnly) rtc->freeMem(pld->send.raysOnly);
        if (pld->send.hitsOnly) rtc->freeMem(pld->send.hitsOnly);
        if (pld->recv.hitsOnly) rtc->freeMem(pld->send.hitsOnly);

        size_t N = ourRequiredQueueSize;
        pld->send.raysOnly = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
        pld->recv.raysOnly = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
        pld->send.hitsOnly = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
        pld->recv.hitsOnly = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
        pld->currentSize = N;
      }
    }
  }

    // step 1: have all ranks exchange which (global) device has how
    // many rays (needed to set up the send/receives)
  void MPIAll2all::exchangeHowManyRaysEachDeviceHas()
  {
    BARNEY_NYI();
  }
  
  void MPIAll2all::traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs)
  {
    ensureAllOurQueuesAreLargeEnough();
    exchangeHowManyRaysEachDeviceHas();

    BARNEY_NYI();
  }

  
}
