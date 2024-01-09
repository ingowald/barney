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

#include "barney/umesh/RTXFieldSampler.h"
#include <owl/owl_device.h>

namespace barney {

  OPTIX_CLOSEST_HIT_PROGRAM(RTXFieldSampler_CH)()
  {}

  OPTIX_INTERSECT_PROGRAM(RTXFieldSampler_IS)()
  {
    const int primID = optixGetPrimitiveIndex();
    const auto &self
      = owl::getProgramData<typename UMeshField::DD>();
    auto &retVal
      = owl::getPRD<float>();

    UMeshField::Element elt = self.elements[primID];
    const vec3f P = optixGetObjectRayOrigin();
    if (self.eltScalar(retVal,elt,P)) {
      optixReportIntersection(0.f, 0);
      optixTerminateRay();
    }
  }
   
}
