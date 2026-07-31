// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    WorkerTopo *topo;
    const Device *device;
    MPIContext *const context;
    struct {
      std::vector<int> rayCounts;
    } global;
    const bool logTopo;
    const bool logQueues;
    const bool opt_mpi;

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

    // only used for opt_mpi variant:
    // struct {
    //   int numRaysReceived;
    // } bothStages;
    struct {
      int sumRaysReceived;
      // opt_mpi only:
      barney_api::mpi::Comm comm;
      std::vector<int> rayCounts;
    } intraNode;
    struct {
      int sumRaysReceived;
      // opt_mpi only:
      barney_api::mpi::Comm comm;
      std::vector<int> rayCounts;
    } crossNodes;
  };

}
