// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/native/globalTrace/RQSBase.h"

namespace BARNEY_NS {
  namespace native {

    struct MPIContext;
  
    struct RQSMPI : public RQSBase
    {
      RQSMPI(MPIContext *context);
    
      /*! forward rays (during global trace); returns true if _after_
        that forward the rays need more tracing (true) or whether
        they're done (false) */
      bool forwardRays(bool needHitIDs) override;
      // int numDifferentModelSlots = -1;
    
      MPIContext *const context;
    };

  }
}

