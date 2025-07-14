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

#pragma once

#include "barney/Context.h"

namespace barney {
  
  struct MPIAll2AllTraceStrategy : public GlobalTraceStrategy
  {
    struct GlobalDevice {
      int rank;
      int local;
      int dgID;
      int island;
    };
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
    struct Island {
      std::vector<int> globalIDs;
    };
    std::vector<GlobalDevice> globalDevices;
    std::vector<Island>       islands;
    std::vector<int>          raysOnGlobalDevice;
    
    MPIAll2AllTraceStrategy(Context *context)
      : GlobalTraceStrategy(context)
    {}
    void resize(int maxRaysPerRayGenOrShadeLaunch) override;
    void traceRays() override;

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
    
