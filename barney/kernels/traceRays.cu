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
#include "owl/owl_device.h"
#include "barney/render/OptixGlobals.h"

// __constant__ barney::render::OptixGlobals optixLaunchParams;
// DECLARE_OPTIX_LAUNCH_PARAMS(barney::render::OptixGlobals);

namespace barney {
  namespace render {
      
    OPTIX_RAYGEN_PROGRAM(traceRays)()
    {
      auto &lp = optixLaunchParams;
      const int rayID
        = owl::getLaunchIndex().x
        + owl::getLaunchDims().x
        * owl::getLaunchIndex().y;

      if (rayID >= lp.numRays)
        return;

      Ray &ray = lp.rays[rayID];

      vec3f dir = ray.dir;
      if (dir.x == 0.f) dir.x = 1e-6f;
      if (dir.y == 0.f) dir.y = 1e-6f;
      if (dir.z == 0.f) dir.z = 1e-6f;

      owl::traceRay(lp.world,
                    owl::Ray(ray.org,
                             dir,
                             0.f,ray.tMax),
                    ray);
    }

  }
}
