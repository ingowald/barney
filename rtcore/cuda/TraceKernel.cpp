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

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/TraceKernel.h"

namespace rtc {
  namespace cuda {

    TraceKernel2D::TraceKernel2D(Device *device,
                                 size_t sizeOfLP,
                                 TraceLaunchFct traceLaunchFct)
      : device(device),
        sizeOfLP(sizeOfLP),
        traceLaunchFct(traceLaunchFct)
    {
      BARNEY_CUDA_CALL(Malloc((void **)&d_lpData,sizeOfLP));
    }

    TraceKernel2D::~TraceKernel2D()
    {
      BARNEY_CUDA_CALL_NOTHROW(Free(d_lpData));
    }

        
    void TraceKernel2D::launch(vec2i launchDims,
                               const void *kernelData)
    {
      SetActiveGPU forDuration(device);   
      BARNEY_CUDA_CALL(MemcpyAsync(d_lpData,kernelData,
                                   sizeOfLP,cudaMemcpyDefault,device->stream));
      traceLaunchFct(device,launchDims,d_lpData);
    }
    
  }
}
