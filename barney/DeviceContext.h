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

#include "barney/render/Ray.h"
#include "barney/render/RayQueue.h"
#include "barney/Camera.h"
#include "barney/DeviceGroup.h"
#include "barney/render/MaterialRegistry.h"
#include "barney/render/SamplerRegistry.h"

namespace barney {

  struct TiledFB;
  struct GlobalModel;
  struct Renderer;
  
  struct DeviceContext
  {
    typedef std::shared_ptr<DeviceContext> SP;
    
    DeviceContext(Device::SP device);

    void shadeRays_launch(Renderer *renderer,
                          GlobalModel *model,
                          TiledFB *fb,
                          int generation);
    void shadeRays_sync();
    void traceRays_launch(GlobalModel *model);
    
    void generateRays_launch(TiledFB *fb,
                             const Camera::DD &camera,
                             int rngSeed);
    void generateRays_sync();

    static OWLParams createLP(Device *device);


    void sync() // const
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_SYNC_CHECK();
    } 
    
    void launch_sync() const
    {
      BARNEY_CUDA_CALL(StreamSynchronize(device->launchStream));
    }

    render::RayQueue rays;
    Device::SP device;
  };
    
}
