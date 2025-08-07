// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/globalTrace/RQSBase.h"

namespace BARNEY_NS {
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

