// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "barney/globalTrace/RQSBase.h"
#include "barney/render/RayQueue.h"

namespace BARNEY_NS {

  RQSBase::PLD *RQSBase::getPLD(Device *device)
  { return &perLogical[device->localRank()]; }

  
  RQSBase::RQSBase(Context *context)
    : GlobalTraceImpl(context),
      perLogical(context->devices->numLogical)
  {
    auto topo = context->topo;
    int islandSize = topo->islandSize();
    for (int local = 0; local < perLogical.size(); local++) {
      auto &pld = perLogical[local];
      int myDev = topo->find(context->myRank(),local);
      int myIsland = topo->islandOf[myDev];
      int myIslandRank = topo->islandRankOf[myDev];

      int myNext = topo->islands[myIsland]
        [(myIslandRank+1) % islandSize];
      int myPrev = topo->islands[myIsland]
        [(myIslandRank+islandSize-1) % islandSize];
      
      pld.sendPartner = &topo->allDevices[myNext];
      pld.recvPartner = &topo->allDevices[myPrev];
    }
  }


  void RQSBase::traceRays(GlobalModel *model,
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

}
