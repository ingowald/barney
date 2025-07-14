// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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
