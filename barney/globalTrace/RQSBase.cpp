// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    bool minimizeNetworkHops = FromEnv::enabled("opt_mpi");
    for (int local = 0; local < perLogical.size(); local++) {
      auto &pld = perLogical[local];
      int myDev = topo->find(context->myRank(),local);
      int myIsland = topo->islandOf[myDev];
      int myIslandRank = topo->islandRankOf[myDev];
      int myNext=-1;
      int myPrev=-1;
      
      if (minimizeNetworkHops) {
        /* make a list of all devices that are in the same island as
           we are - store that in a tuple with device gid last, and
           host and gpu index is tuple slots 0 and 1, respectively,
           such that we can then sort by order host->gpu->devID */
        std::vector<std::tuple</*host*/int,/*gpu*/int,/*devID*/int>> devsInIsland;
        for (int devID=0; devID<topo->allDevices.size(); devID++) {
          auto dev = topo->allDevices[devID];
          if (topo->islandOf[devID] == myIsland)
            devsInIsland.push_back({
                topo->physicalHostIndexOf[devID],
                topo->physicalGpuIndexOf[devID],
                devID});
        }
        assert(devsInIsland.size() == islandSize);
        /* sort the devsInIsland tuples. after this sort we should
           have a list of all devices that is sorted such thatwe have
           all those device on host 0 first, etc; and those on the
           same host are sroted by gpu ID. In this order, devices that
           ARE on the same host will appear next to each other in that
           sorted list. */
        std::sort(devsInIsland.begin(),devsInIsland.end());
        int myIdx = 0;
        while (std::get<2>(devsInIsland[myIdx]) != myDev) ++myIdx;
        myNext = std::get<2>(devsInIsland[(myIdx+1)%islandSize]);
        myPrev = std::get<2>(devsInIsland[(myIdx+islandSize-1)%islandSize]);
      } else {
        myNext = topo->islands[myIsland]
          [(myIslandRank+1) % islandSize];
        myPrev = topo->islands[myIsland]
          [(myIslandRank+islandSize-1) % islandSize];
      }
      
      pld.myDev       = &topo->allDevices[myDev];
      pld.sendPartner = &topo->allDevices[myNext];
      pld.recvPartner = &topo->allDevices[myPrev];
    }
  
    if (FromEnv::get()->logTopo) {
      std::stringstream ss;
      for (int localIdx=0;localIdx<context->devices->size();localIdx++) {
        auto device = context->devices->get(localIdx);
        auto pld = getPLD(device);
        int gid = pld->myDev->gid;
        ss << "#bn.rqs(" << context->myRank() << "." << localIdx << "):"
           << " island " << topo->islandOf[gid]
           << "\n device { " << topo->toString(gid) << "}"
           << "\n sendsTo { " << topo->toString(pld->sendPartner->gid) << " }" 
           << "\n recvsFrom { " << topo->toString(pld->recvPartner->gid) << " }" 
           << "\n";
      }
      std::cout << ss.str();
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
