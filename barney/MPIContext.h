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
#include "barney/MPIWrappers.h"

namespace barney {

  /*! barney context for collaborative MPI-parallel rendering */
  struct MPIContext : public Context
  {
    MPIContext(const mpi::Comm &comm,
               const std::vector<int> &dataGroupIDs,
               const std::vector<int> &gpuIDs);

    /*! create a frame buffer object suitable to this context */
    FrameBuffer *createFB(int owningRank) override;

    void render(Model *model,
                const BNCamera *camera,
                FrameBuffer *fb) override;

    /*! gives, for a given worker rank, the rank that this same rank
        has in the parent 'world' communicator */
    std::vector<int> worldRankOfWorker;
    std::vector<int> workerRankOfWorldRank;
    
    int gpusPerWorker;
    
    mpi::Comm world;
    mpi::Comm workers;
    int numWorkers;
  };

}
