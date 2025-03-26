// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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
#include "barney/common/MPIWrappers.h"

namespace BARNEY_NS {

  /*! barney context for collaborative MPI-parallel rendering */
  struct MPIContext : public Context
  {
    MPIContext(const barney_api::mpi::Comm &worldComm,
               const barney_api::mpi::Comm &workersComm,
               bool isActiveWorker,
               const std::vector<int> &dataGroupIDs,
               const std::vector<int> &gpuIDs);

    /*! create a frame buffer object suitable to this context */
    std::shared_ptr<barney_api::FrameBuffer>
    createFrameBuffer(int owningRank) override;

    void render(Renderer    *renderer,
                GlobalModel *model,
                const Camera::DD &camera,
                FrameBuffer *fb) override;

    /*! gives, for a given worker rank, the rank that this same rank
        has in the parent 'world' communicator */
    std::vector<int> worldRankOfWorker;
    std::vector<int> workerRankOfWorldRank;

    /*! forward rays (during global trace); returns if _after_ that
        forward the rays need more tracing (true) or whether they're
        done (false) */
    bool forwardRays(bool needHitIDs) override;

    // for debugging ...
    void barrier(bool warn=true) override {
      if (warn) PING;
      workers.barrier();
      if (warn) usleep(100);
    }
    

    /*! returns how many rays are active in all ray queues, across all
        devices and, where applicable, across all ranks */
    int numRaysActiveGlobally() override;
    
    int gpusPerWorker;
    int numDifferentModelSlots = -1;
    int numTimesForwarded = 0;
    
    barney_api::mpi::Comm world;
    barney_api::mpi::Comm workers;
    int numWorkers;
  };

}
