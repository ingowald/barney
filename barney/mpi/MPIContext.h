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
#include "barney/mpi/MPIWrappers.h"

namespace barney {

  struct DistFB : public FrameBuffer {
    enum { tileSize = 32 };
    
    typedef std::shared_ptr<DistFB> SP;

    static SP create()
    { return std::make_shared<DistFB>(); }
    
    void resize(vec2i size) override { PING; }

    int numTiles = 0;
  };
  
  struct MPIContext : public Context
  {
    MPIContext(const mpi::Comm &comm,
               const std::vector<int> &dataGroupIDs,
               const std::vector<int> &gpuIDs);

    FrameBuffer *createFB() override
    { return initReference(DistFB::create()); }
    
    mpi::Comm comm;
  };

}
