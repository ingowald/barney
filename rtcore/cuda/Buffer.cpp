// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Buffer.h"
#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace cuda {
    
    Buffer::Buffer(Device *device,
                   size_t numBytes,
                   const void *initValues)
      : device(device)
    {
      if (numBytes == 0) return;
      
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Malloc((void**)&d_data,numBytes));
      if (initValues)
        BARNEY_CUDA_CALL(Memcpy(d_data,initValues,numBytes,cudaMemcpyDefault));
    }

    Buffer::~Buffer()
    {
      if (!d_data) return;
      BARNEY_CUDA_CALL_NOTHROW(Free(d_data));
    }
    
    void Buffer::upload(const void *data, size_t numBytes, size_t offset)
    {
      if (!d_data) return;
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Memcpy(((char *)d_data)+offset,data,numBytes,cudaMemcpyDefault));
    }

    void Buffer::resize(size_t numBytes)
    {
      SetActiveGPU forDuration(device);
      if (d_data) 
        BARNEY_CUDA_CALL(Free(d_data));
      BARNEY_CUDA_CALL(Malloc((void**)&d_data,numBytes));
    }
    
  }
}
