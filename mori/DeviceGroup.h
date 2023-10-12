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

#pragma once

#include "mori/common.h"
#include "owl/owl.h"

namespace mori {

  struct DeviceContext {
    DeviceContext(int gpuID);
    
    void sync() const
    {
      MORI_CUDA_CALL(StreamSynchronize(stream));
    }

    OWLContext   owl;
    int          gpuID;
    cudaStream_t stream;
    int          tileIndexOffset = 0;
    int          tileIndexScale  = 0;
  };
  
  /*! stolen from owl/DeviceContext: helper class that will set the
      active cuda device (to the device associated with a given
      Context::DeviceData) for the duration fo the lifetime of this
      class, and resets it to whatever it was after class dies */
  struct SetActiveGPU {
    inline SetActiveGPU(const DeviceContext *device)
    {
      assert(device);
      MORI_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      MORI_CUDA_CHECK(cudaSetDevice(device->gpuID));
    }
    
    inline SetActiveGPU(int cudaDeviceID)
    {
      MORI_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      MORI_CUDA_CHECK(cudaSetDevice(cudaDeviceID));
    }
    inline ~SetActiveGPU()
    {
      MORI_CUDA_CHECK_NOTHROW(cudaSetDevice(savedActiveDeviceID));
    }
  private:
    int savedActiveDeviceID = -1;
  };
  

  // still need this?
  struct DeviceGroup {
    typedef std::shared_ptr<DeviceGroup> SP;

    static SP create(const std::vector<int> &gpuIDs)
    { return std::make_shared<DeviceGroup>(gpuIDs); }
    
    DeviceGroup(const std::vector<int> &gpuIDs);

  };
  
}
