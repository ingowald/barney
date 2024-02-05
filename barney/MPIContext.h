// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

namespace barney {

  /*! barney context for collaborative MPI-parallel rendering */
  struct MPIContext : public Context
  {
    MPIContext(const mpi::Comm &worldComm,
               const mpi::Comm &workersComm,
               bool isActiveWorker,
               const std::vector<int> &dataGroupIDs,
               const std::vector<int> &gpuIDs);

    /*! create a frame buffer object suitable to this context */
    FrameBuffer *createFB(int owningRank) override;

    void render(Model *model,
                const Camera &camera,
                FrameBuffer *fb,
                int pathsPerPixel) override;

    /*! gives, for a given worker rank, the rank that this same rank
        has in the parent 'world' communicator */
    std::vector<int> worldRankOfWorker;
    std::vector<int> workerRankOfWorldRank;

    /*! forward rays (during global trace); returns if _after_ that
        forward the rays need more tracing (true) or whether they're
        done (false) */
    bool forwardRays() override;

    // for debugging ...
    void barrier(bool warn=true) override { if (warn) PING; workers.barrier(); usleep(100); }
    

    /*! returns how many rays are active in all ray queues, across all
        devices and, where applicable, across all ranks */
    int numRaysActiveGlobally() override;
    
    int gpusPerWorker;
    int numDifferentDataGroups = -1;
    int numTimesForwarded = 0;
    
    mpi::Comm world;
    mpi::Comm workers;
    int numWorkers;
  };

}
