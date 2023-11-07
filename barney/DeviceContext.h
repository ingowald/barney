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

#include "barney/Ray.h"
#include "barney/Camera.h"

namespace barney {

  struct TiledFB;
  struct Model;
  
  struct DeviceContext
  {
    typedef std::shared_ptr<DeviceContext> SP;
    
    /*! this is the device data for the launch params */
    struct DD {
      OptixTraversableHandle world;
      Ray                   *rays;
      int                    numRays;
    };
    
    DeviceContext(Device::SP device);

    void shadeRays_launch(TiledFB *fb, int generation);
    void shadeRays_sync();
    void traceRays_launch(Model *model);
    
    void generateRays_launch(TiledFB *fb,
                             const Camera &camera,
                             int rngSeed);
    void generateRays_sync();

    static OWLParams createLP(Device *device);


    void sync() const {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_SYNC_CHECK();
    }
    
    void launch_sync() const
    {
      BARNEY_CUDA_CALL(StreamSynchronize(device->launchStream));
    }

    // OWLLaunchParams    const lp;
    // /*! this is the stream (from the *launch params*) for all *launch*
    //     related operations */
    // cudaStream_t       const launchStream;
    
    barney::RayQueue rays;
    /*! each barneycontext gets its own LP: even though that lp's
        context is (possibly) shared across multiple device contextes
        (and thus, across multiple barney contexts) well still have one
        LP for each device/barney context. thus, we'll have a separate
        stream for each device/barney context */
    Device::SP device;
  };
    
}
