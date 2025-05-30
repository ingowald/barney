// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

/* kernelInterface - how device-side compute kernels (e.g., shadeRays,
   but not programs like intersect or anyhit) can talk to rtcore */

#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/ComputeInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif

#if BARNEY_RTC_OPTIX
# include "optix/ComputeInterface.h"
namespace rtc { using namespace rtc::optix; }
#endif

#if BARNEY_RTC_EMBREE
# include "embree/ComputeInterface.h"
namespace rtc { using namespace rtc::embree; }
#endif
