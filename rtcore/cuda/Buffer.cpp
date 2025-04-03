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
      PING; PRINT(numBytes); PRINT(d_data);
      if (initValues)
        BARNEY_CUDA_CALL(Memcpy(d_data,initValues,numBytes,cudaMemcpyDefault));
    }

    Buffer::~Buffer()
    {
      if (!d_data) return;
      PING; PRINT(d_data);
      BARNEY_CUDA_CALL_NOTHROW(Free(d_data));
    }
    
    void Buffer::upload(const void *data, size_t numBytes, size_t offset)
    {
      if (!d_data) return;
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(Memcpy(((char *)d_data)+offset,data,numBytes,cudaMemcpyDefault));
    }

  }
}
