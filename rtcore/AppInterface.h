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

/*! \file rtcore/AppInterface.h Defines how the app/host side of
    barney can talk to rtcore; e.g., to create new groups, geometries,
    etc */

#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/AppInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif

#if BARNEY_RTC_OPTIX
# include "optix/AppInterface.h"
namespace rtc { using namespace rtc::optix; }
#endif

#if BARNEY_RTC_EMBREE
# include "embree/AppInterface.h"
namespace rtc { using namespace rtc::embree; }
#endif
