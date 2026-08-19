// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    /*! a barney context for "local"-node rendering - no MPI */
    struct LocalContext : public Context {
    
      LocalContext(const std::vector<DataGroupDescriptor> &dataGroupsOnThisContext);

      virtual ~LocalContext();

      static WorkerTopo::SP
      makeTopo(const std::vector<DataGroupDescriptor> &localDataGroups);
      
      static Context *create(/*! this is the NUMBER of different data
                               groups/slots inthat context. a data
                               *group* is one or more GPUs that share a
                               common set of geometric/volumetric
                               objects. */
                             int        numDataGroupsOnThisContext,
                             /*! tells which data rank / 'color' of data
                               will be stored in each of the
                               `numDataGroupsOnThisContext` data groups
                               of this context. This array *must* have
                               `numDataGroupsOnThisContext` entries */
                             const int *dataRanksInDataGroup,
                             /*! which gpu(s) to use for this
                               process. GPUs will be assigned to data
                               groups on a round-robin basis, so the i'th
                               GPU listed here will get assigned to data
                               group 'i%numDataGroupsOnThisContext' */
                             int numGPUs,
                             const int *gpuIDs);
    
      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "LocalFB{}"; }

      int numRaysActiveGlobally() override;
    
      void render(Renderer    *renderer,
                  GlobalModel *model,
                  Camera      *camera,
                  FrameBuffer *fb) override;

      int myRank() override { return 0; }
      int mySize() override { return 1; }

      /*! create a frame buffer object suitable to this context */
      std::shared_ptr<FrameBuffer>
      createFrameBuffer() override;
    };

  }
}
