// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Context.h"

namespace BARNEY_NS {

  /*! a barney context for "local"-node rendering - no MPI */
  struct LocalContext : public Context {
    
    LocalContext(const std::vector<LocalSlot> &localSlots);

    virtual ~LocalContext();

    static WorkerTopo::SP makeTopo(const std::vector<LocalSlot> &localSlots);
    
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
    std::shared_ptr<barney_api::FrameBuffer>
    createFrameBuffer() override;
  };
}
