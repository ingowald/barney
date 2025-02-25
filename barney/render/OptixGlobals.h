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

#include "barney/DeviceContext.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/render/Sampler.h"
#include "barney/render/HitAttributes.h"

namespace BARNEY_NS {
  namespace render {

    struct OptixGlobals {
#if BARNEY_DEVICE_PROGRAM
      static inline __rtc_device
      const OptixGlobals &get(const rtc::TraceInterface &dev);
#endif
      
      int                    numRays;
      int                    globalIndex;
      Sampler::DD           *samplers;
      DeviceMaterial        *materials;
      // OptixTraversableHandle world;
      rtc::device::AccelHandle  world;
      Ray                   *rays;
    };
  }
}

// #ifdef __CUDA_ARCH__
// extern __constant__ BARNEY_NS::render::OptixGlobals optixLaunchParams;
// // # ifndef DECLARE_OPTIX_LAUNCH_PARAMS
// // /*! in owl we can only change the _type_ of launch params, they always
// //     need to be caleld 'optixLaunchParams', and must have __constant__
// //     storage*/
// // #  define DECLARE_OPTIX_LAUNCH_PARAMS(LPType ) \
// //   extern __constant__ LPType optixLaunchParams
// // # endif
// // DECLARE_OPTIX_LAUNCH_PARAMS(barney::render::OptixGlobals);
// // // extern __constant__ barney::render::OptixGlobals optixLaunchParams;
// #endifS

namespace BARNEY_NS {
  namespace render {

#if BARNEY_DEVICE_PROGRAM
    inline __rtc_device
    const OptixGlobals &OptixGlobals::get(const rtc::TraceInterface &be)
    {
      return *(OptixGlobals*)be.getLPData();
    }
#endif
  }
}
