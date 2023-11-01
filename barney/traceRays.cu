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

#include "barney/DeviceContext.h"
#include "owl/owl_device.h"

__constant__ struct {
  __forceinline__ __device__ const barney::DeviceContext::DD &get() const
  { return *(const barney::DeviceContext::DD*)this; }
  
  float4 podData[(sizeof(barney::DeviceContext::DD)+sizeof(float4)-1)/sizeof(float4)];
} optixLaunchParams;

namespace barney {
  
  OPTIX_RAYGEN_PROGRAM(traceRays)()
  {
    auto &lp = optixLaunchParams.get();
    const int rayID
      = owl::getLaunchIndex().x
      + owl::getLaunchDims().x
      * owl::getLaunchIndex().y;
    
    if (rayID >= lp.numRays)
      return;

    Ray &ray = lp.rays[rayID];
    owl::traceRay(lp.world,
                  owl::Ray(ray.org,
                           ray.dir,
                           0.f,ray.tMax),
                  ray);
  }

}
