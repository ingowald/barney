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

#include "barney/DeviceContext.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/render/Sampler.h"
#include "barney/render/HitAttributes.h"

namespace barney {
  namespace render {
      
    struct OptixGlobals {
      static inline __device__ const OptixGlobals &get();
      
      Sampler::DD           *samplers;
      DeviceMaterial        *materials;
      OptixTraversableHandle world;
      Ray                   *rays;
      int                    numRays;
    };
    
  }
}

#ifdef __CUDA_ARCH__
# ifndef DECLARE_OPTIX_LAUNCH_PARAMS
/*! in owl we can only change the _type_ of launch params, they always
    need to be caleld 'optixLaunchParams', and must have __constant__
    storage*/
#  define DECLARE_OPTIX_LAUNCH_PARAMS(a) extern __constant__ a optixLaunchParams
# endif
DECLARE_OPTIX_LAUNCH_PARAMS(barney::render::OptixGlobals);
// extern __constant__ barney::render::OptixGlobals optixLaunchParams;
#endif

namespace barney {
  namespace render {
      
#ifdef __CUDA_ARCH__
    inline __device__ const OptixGlobals &OptixGlobals::get()
    { return optixLaunchParams; }
#endif      
  }
}
