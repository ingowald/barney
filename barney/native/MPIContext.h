// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/native/Context.h"
#include "barney/native/common/MPIWrappers.h"

namespace BARNEY_NS {
  namespace native {
    
    struct FrameBuffer;
    
    /*! barney context for collaborative MPI-parallel rendering */
    struct MPIContext : public Context
    {
      MPIContext(const Comm &worldComm,
                 const Comm &workersComm,
                 const std::vector<DataGroupDescriptor> &dataGroupsOnThisContext);
      virtual ~MPIContext();
    
      static WorkerTopo::SP
      makeTopo(const Comm &worldComm,
               const Comm &workersComm,
               const std::vector<DataGroupDescriptor> &dataGroups);
    
      /*! create a frame buffer object suitable to this context */
      std::shared_ptr<FrameBuffer>
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
      void barrier(bool warn=true) override;

      /*! returns how many rays are active in all ray queues, across all
        devices and, where applicable, across all ranks */
      int numRaysActiveGlobally() override;

      int myRank() override { return world.rank; }
      int mySize() override { return world.size; }
    
      int gpusPerWorker;

      int numWorkers() const { return workers.size; }
    
      Comm world;
      Comm workers;
      // int numWorkers;
    };

  }
}
