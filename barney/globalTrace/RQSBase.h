// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/Context.h"

namespace BARNEY_NS {

    /*! implements global tracing using ray queue cycling, but leaving
      it abstract how exactly a ray queue is cycled to another device
      (allowing to instantiate that for both cudamemcpy and mpi
      send/recv */
  struct RQSBase : public GlobalTraceImpl
  {
    struct PLD {
      const WorkerTopo::Device *sendPartner;
      const WorkerTopo::Device *recvPartner;
    };
    PLD *getPLD(Device *device);

    std::vector<PLD> perLogical;
    
    RQSBase(Context *context);

    void traceRays(GlobalModel *model,
                   uint32_t rngSeed,
                   bool needHitIDs) override;

    /*! forward rays (during global trace); returns if _after_ that
      forward the rays need more tracing (true) or whether they're
      done (false) */
    virtual bool forwardRays(bool needHitIDs) = 0;

    int numTimesForwarded = 0;
  };
  
}

