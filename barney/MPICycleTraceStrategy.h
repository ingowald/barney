// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "barney/LocalCycleTraceStrategy.h"

namespace BARNEY_NS {
  struct MPIContext;
  
  struct MPICycleTraceStrategy : public RayQueueCycleTraceStrategy
  {
    MPICycleTraceStrategy(MPIContext *context);
    
    /*! forward rays (during global trace); returns true if _after_
      that forward the rays need more tracing (true) or whether
      they're done (false) */
    bool forwardRays(bool needHitIDs) override;
    // int numDifferentModelSlots = -1;
    
    MPIContext *const context;
  };
      
}

