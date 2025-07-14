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

namespace BARNEY_NS {

  /*! implements global tracing using ray queue cycling, but leaving
      it abstract how exactly a ray queue is cycled to another device
      (allowing to instantiate that for both cudamemcpy and mpi
      send/recv */
  struct RayQueueCycleTraceStrategy : public GlobalTraceStrategy
  {
    struct PLD {
      const WorkerTopo::Device *sendPartner;
      const WorkerTopo::Device *recvPartner;
    };
    PLD *getPLD(Device *device);

    std::vector<PLD> perLogical;
    
    RayQueueCycleTraceStrategy(Context *context)
      : GlobalTraceStrategy(context)
    {}

    void resize(int maxRaysPerRayGenOrShadeLaunch) override;
    void traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs) override;

    /*! forward rays (during global trace); returns if _after_ that
      forward the rays need more tracing (true) or whether they're
      done (false) */
    virtual bool forwardRays(bool needHitIDs) = 0;

    int numTimesForwarded = 0;
  };

  /*! implements queue cycling for a local context only, where we can
    use memcpy to move rays between different local devices */
  struct LocalCycleTraceStrategy : public RayQueueCycleTraceStrategy
  {
    LocalCycleTraceStrategy(Context *context);
    
    /*! forward rays (during global trace); returns if _after_ that
        forward the rays need more tracing (true) or whether they're
        done (false) */
    bool forwardRays(bool needHitIDs) override;
  };
      
}

