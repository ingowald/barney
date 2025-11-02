// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Malloc((void **)&d_lpData,sizeOfLP));
    }

    TraceKernel2D::~TraceKernel2D()
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL_NOTHROW(Free(d_lpData));
    }

        
    void TraceKernel2D::launch(vec2i launchDims,
                               const void *kernelData)
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(MemcpyAsync(d_lpData,kernelData,
                                   sizeOfLP,cudaMemcpyDefault,device->stream));
      // device->sync();
      traceLaunchFct(device,launchDims,d_lpData);
    }
    
  }
}
