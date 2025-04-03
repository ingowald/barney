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

#pragma once

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace cuda {
    
    struct Device;

    typedef void (*TraceLaunchFct)(Device *device, vec2i dims, const void *lpData);
    
    struct TraceKernel2D {
      TraceKernel2D(Device *device,
                    size_t sizeOfLP,
                    TraceLaunchFct traceLaunchFct);
      ~TraceKernel2D();
      void launch(vec2i launchDims,
                  const void *kernelData);
      
      Device        *const device;
      TraceLaunchFct const traceLaunchFct;
      size_t         const sizeOfLP;
      void          *d_lpData = 0;
    };

  }
}

#define RTC_IMPORT_TRACE2D(fileNameBase,name,sizeOfLP)          \
  void rtc_cuda_launch_##name(rtc::Device *device,              \
                              vec2i dims,                       \
                              const void *lpData);              \
                                                                \
  ::rtc::TraceKernel2D *createTrace_##name(rtc::Device *device) \
  {                                                             \
    return new ::rtc::cuda::TraceKernel2D                       \
      (device,sizeOfLP,rtc_cuda_launch_##name);                 \
  }                                                             \
    
