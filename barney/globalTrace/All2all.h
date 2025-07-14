// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/Context.h"
#include "barney/render/Ray.h"

namespace BARNEY_NS {

  using render::RayOnly;
  using render::HitOnly;
  
  struct MPIAll2all : public GlobalTraceImpl
  {
    struct PLD {
      struct {
        RayOnly *raysOnly = 0;
        HitOnly *hitsOnly = 0;
      } send, recv;
      struct {
        std::vector<int> rayCount;
        std::vector<int> rayOffset;
      } perIslandPeer;
      int numRemoteRaysReceived;
    };
    
    MPIAll2all(Context *context);
    void resize(int maxRaysPerRayGenOrShadeLaunch) override;
    void traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs) override;

    // ====================== helper fcts ======================

    // step 1: have all ranks exchange which (global) device has how
    // many rays (needed to set up the send/receives)
    void exchangeHowManyRaysEachDeviceHas();
    
    // step 2: 'broadcast' all rays to resp every other (global)
    // device within the same island as the broadcasting device. at
    // end of this stage every device shold have a local copy of all
    // the rays that any other device on the same island has.
    void stageLocalRaySends();
    void stageRemoteRayReceives();
    void executeRaySendsAndReceives();

    // step 3: trace all rays on each device
    void stageReceivedRaysIntoLocalRayQueues();
    void traceAllReceivedRaysInLocalRayQueues();

    // step 4: send all traced hits back to devices that originally
    // sent the rays
    void stageLocalHitSends();
    void stageRemoteHitReceives();
    void executeHitSendsAndReceives();

    // step 5: merge all the received hits back with the rays that
    // spawend them, and write them into local ray queue.
    void mergeReceivedHitsWithOriginalRays();
  };
}


#if SINGLE_CYCLE_RQS
                        , int maxRaysAcrossAllRanks
#endif
#if SINGLE_CYCLE_RQS
    if (rqs.raysOnly) rtc->freeMem(rqs.raysOnly);
    if (rqs.hitsOnly) rtc->freeMem(rqs.hitsOnly);
    rqs.raysOnly = (RayOnly*)rtc->allocMem(maxRaysAcrossAllRanks*sizeof(RayOnly));
    rqs.hitsOnly = (HitOnly*)rtc->allocMem(maxRaysAcrossAllRanks*sizeof(HitOnly));
#endif
    
