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

/*! \file rtcore/cudaCommon/ComputeKerne.h Defines basic abstraction
  for 1D, 2D, and 3D compute kernels, and the IMPORT macros to
  import such kernels on host code */

#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace cuda_common {

    struct ComputeKernel1D {
      virtual void launch(unsigned int nb, unsigned int bs,
                          const void *pKernelData) = 0;
    };
    
    struct ComputeKernel2D {
      virtual void launch(vec2ui nb, vec2ui bs,
                          const void *pKernelData) = 0;
    };
    
    struct ComputeKernel3D {
      virtual void launch(vec3ui nb, vec3ui bs,
                          const void *pKernelData) = 0;
    };
    
  }
}

#define RTC_IMPORT_COMPUTE1D(name)                                      \
    rtc::ComputeKernel1D *createCompute_##name(rtc::Device *dev);       
#define RTC_IMPORT_COMPUTE2D(name)                                      \
    rtc::ComputeKernel2D *createCompute_##name(rtc::Device *dev);       
#define RTC_IMPORT_COMPUTE3D(name)                                      \
    rtc::ComputeKernel3D *createCompute_##name(rtc::Device *dev);       


