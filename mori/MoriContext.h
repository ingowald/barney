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

#include "mori/Ray.h"
#include "mori/TiledFB.h"
#include "mori/Camera.h"

namespace mori {

  
  struct MoriContext : public DeviceContext
  {
    MoriContext() : rays(this) {}

    void shadeRays_launch(TiledFB *fb);

    void  generateRays_launch(TiledFB *fb,
                              const Camera &camera,
                              int rngSeed);
    void  generateRays_sync();
    
    mori::RayQueue rays;
  };
    
}
