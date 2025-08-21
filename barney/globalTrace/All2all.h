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
  
  struct MPIAll2all : public GlobalTraceImpl
  {
    struct PLD {
      struct {
        RayOnly *raysOnly = 0;
        HitOnly *hitsOnly = 0;
      } send, recv;
      Ray *stagedRayQueue = 0;
      Ray *savedOriginalRayQueue;
      int  savedOriginalRayCount;
      int currentSize = 0;
      struct {
        std::vector<int> rayCount;
        // std::vector<int> rayOffset;
      } perIslandPeer;
      int numRemoteRaysReceived;
    };
    
    PLD *getPLD(Device *device);

    std::vector<PLD> perLogical;
    
    MPIAll2all(MPIContext *context);
    void traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs) override;

    // ====================== helper fcts ======================

    // step 1: have all ranks exchange which (global) device has how
    // many rays (needed to set up the send/receives)
    void exchangeHowManyRaysEachDeviceHas();
    
    // step 2: 'broadcast' all rays to resp every other (global)
    // device within the same island as the broadcasting device. at
    // end of this stage every device shold have a local copy of all
    // the rays that any other device on the same island has.
    void sendAndReceiveRays();

    // step 3: trace all rays on each device
    void traceAllReceivedRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs);

    // step 4: send all traced hits back to devices that originally
    // sent the rays, and correspondingly receive those from others
    void sendAndReceiveHits();

    // step 5: merge all the received hits back with the rays that
    // spawend them, and write them into local ray queue.
    void mergeReceivedHitsWithOriginalRays();

    // maintenance: make sure that our own queus are big enough for
    // whatever ray queues barney local uses.
    void ensureAllOurQueuesAreLargeEnough();
    MPIContext *const context;
    // if set, we'll do all sends/recvs within a single mpi_all2all
    // call, instead of doing N indiviusal isends and N
    // indiv. irecvs. only works for one gpu per rank.
    bool opt_mpi;
  };

}


