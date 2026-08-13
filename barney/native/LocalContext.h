// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    /*! a barney context for "local"-node rendering - no MPI */
    struct LocalContext : public Context {
    
      LocalContext(const std::vector<LocalSlot> &localSlots);

      virtual ~LocalContext();

      static WorkerTopo::SP makeTopo(const std::vector<LocalSlot> &localSlots);
      static Context *create(/*! how many data slots this context is to
                               offer, and which part(s) of the
                               distributed model data these slot(s)
                               will hold */
                             const int *dataRanksOnThisContext,
                             int        numDataRanksOnThisContext,
                             /*! which gpu(s) to use for this
                               process. default is to distribute
                               node's GPUs equally over all ranks on
                               that given node */
                             const int *gpuIDs,
                             int  numGPUs);
    
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
