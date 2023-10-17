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
#include "mori/Camera.h"

namespace mori {

  struct TiledFB;
  
  struct MoriContext
  {
    typedef std::shared_ptr<MoriContext> SP;
    
    /*! this is the device data for the launch params */
    struct DD {
      OptixTraversableHandle world;
      RayQueue::DD rayQueue;
    };
    
    MoriContext(Device::SP device);

    void shadeRays_launch(TiledFB *fb);
    
    void generateRays_launch(TiledFB *fb,
                             const Camera &camera,
                             int rngSeed);
    void generateRays_sync();

    static OWLParams createLP(Device *device);

    
    void launch_sync() const
    {
      MORI_CUDA_CALL(StreamSynchronize(launchStream));
    }

    OWLLaunchParams    const lp;
    /*! this is the stream (from the *launch params*) for all *launch*
        related operations */
    cudaStream_t       const launchStream;
    
    mori::RayQueue rays;
    /*! each moricontext gets its own LP: even though that lp's
        context is (possibly) shared across multiple device contextes
        (and thus, across multiple mori contexts) well still have one
        LP for each device/mori context. thus, we'll have a separate
        stream for each device/mori context */
    Device::SP device;
  };
    
}
