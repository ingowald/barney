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

namespace barney {

  /*! a barney context for "local"-node rendering - no MPI */
  struct LocalContext : public Context {
    
    LocalContext(const std::vector<int> &dataGroupIDs,
                 const std::vector<int> &gpuIDs);

    ~LocalContext() { PING; PING; fflush(0); }

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "LocalFB{}"; }

    void render(Model *model,
                const Camera *camera,
                FrameBuffer *fb) override;

    /*! forward rays (during global trace); returns if _after_ that
        forward the rays need more tracing (true) or whether they're
        done (false) */
    bool forwardRays() override;

    /*! returns how many rays are active in all ray queues, across all
        devices and, where applicable, across all ranks */
    int numRaysActiveGlobally() override;
    
    
    /*! create a frame buffer object suitable to this context */
    FrameBuffer *createFB(int owningRank) override;
  };
}
