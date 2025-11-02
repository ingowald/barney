// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <unistd.h>
#include "barney/Context.h"
#include "barney/common/MPIWrappers.h"

namespace BARNEY_NS {

  /*! barney context for collaborative MPI-parallel rendering */
  struct MPIContext : public Context
  {
    MPIContext(const barney_api::mpi::Comm &worldComm,
               const barney_api::mpi::Comm &workersComm,
               const std::vector<LocalSlot> &localSlots,
               bool userSuppliedGpuListWasEmpty);
    virtual ~MPIContext();
    
    static WorkerTopo::SP makeTopo(const barney_api::mpi::Comm &worldComm,
                                   const barney_api::mpi::Comm &workersComm,
                                   const std::vector<LocalSlot> &localSlots);
    
    /*! create a frame buffer object suitable to this context */
    std::shared_ptr<barney_api::FrameBuffer>
    createFrameBuffer() override;

    void render(Renderer    *renderer,
                GlobalModel *model,
                Camera      *camera,
                FrameBuffer *fb) override;

    /*! gives, for a given worker rank, the rank that this same rank
        has in the parent 'world' communicator */
    // std::vector<int> worldRankOfWorker;
    // std::vector<int> workerRankOfWorldRank;

    // for debugging ...
    void barrier(bool warn=true) override {
      if (warn) PING;
      workers.barrier();
      if (warn) usleep(100);
    }
    

    /*! returns how many rays are active in all ray queues, across all
        devices and, where applicable, across all ranks */
    int numRaysActiveGlobally() override;

    int myRank() override { return world.rank; }
    int mySize() override { return world.size; }
    
    int gpusPerWorker;

    int numWorkers() const { return workers.size; }
    
    barney_api::mpi::Comm world;
    barney_api::mpi::Comm workers;
    // int numWorkers;
  };

}
