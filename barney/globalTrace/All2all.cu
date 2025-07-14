// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "barney/globalTrace/All2all.h"

namespace BARNEY_NS {

  MPIAll2all::MPIAll2all(Context *context)
    : GlobalTraceImpl(context)
  {}

  void MPIAll2all::resize(int maxRaysPerRayGenOrShadeLaunch)
  {
    BARNEY_NYI();
  }
  
  void MPIAll2all::traceRays(GlobalModel *model, uint32_t rngSeed, bool needHitIDs)
  {
    BARNEY_NYI();
  }
  
  
}
