// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/Context.h"
#include "barney/render/Ray.h"

namespace BARNEY_NS {

  struct MPIContext;
  using render::RayOnly;
  using render::HitOnly;
  using render::Ray;

  /*! for now, only implmenet for
    a) single device per rank
    b) single island
    c) fixed number of ranks per physical host
  */
  struct TwoStage : public GlobalTraceImpl {
    int hostIdx;
    int gpuIdx;
    int myGID;

    int gpusPerHost;
    int numGlobal;
    int numHosts;
    
    RayOnly *raysOnly[2] = { 0, 0 };
    HitOnly *hitsOnly[2] = { 0, 0 };
    int currentReservedSize = 0;
    Ray *stagedRayQueue = 0;
    // Ray *savedOriginalRayQueue;
    // int  savedOriginalRayCount;
    struct {
      int numRaysReceived;
    } intraNodes, bothStages;
    WorkerTopo *topo;
    const Device *device;
    MPIContext *const context;
    std::vector<int> rayCounts;

    TwoStage(MPIContext *context);
    void traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs) override;
    
    void ensureAllOurQueuesAreLargeEnough();
    void exchangeHowManyRaysEachDeviceHas();    
    void sendAndReceiveRays_crossNodes();
    void sendAndReceiveRays_intraNode();
    void exchangeHits_crossNodes();
    void exchangeHits_intraNode();
    
    void reduceHits_intraNode();
    void reduceHits_crossNodes();

    int rankOf(int hostIdx, int gpuIdx)
    { return _rankOf[hostIdx*gpusPerHost+gpuIdx]; }
    std::vector<int> _rankOf;

    // step 3: trace all rays on each device
    void traceAllReceivedRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs);

    barney_api::mpi::Comm &world;
  };

}
