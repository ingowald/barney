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

  struct Device {
    typedef std::shared_ptr<Device> SP;
    
    Device(int gpuID,
           int globalIndex,
           int globalIndexStep);

    ~Device()
    { printf("MORI DEVICE IS DYING\n"); }
    
    std::mutex               mutex;
    int                const cudaID;
    OWLContext         const owlContext;
    cudaStream_t       const nonLaunchStream;
    int                const globalIndex;
    int                const globalIndexStep;
    
    std::map<std::string,OWLGeomType> geomTypes;
    OWLGeomType getOrCreateGeomTypeFor(const std::string &geomTypeString,
                                       OWLGeomType (*createOnce)(Device *));
  };
  
  /*! stolen from owl/Device: helper class that will set the
      active cuda device (to the device associated with a given
      Context::DeviceData) for the duration fo the lifetime of this
      class, and resets it to whatever it was after class dies */
  struct SetActiveGPU {
    inline SetActiveGPU(const Device *device)
    {
      assert(device);
      MORI_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      MORI_CUDA_CHECK(cudaSetDevice(device->cudaID));
    }
    inline SetActiveGPU(const Device::SP &device)
    {
      assert(device);
      MORI_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      MORI_CUDA_CHECK(cudaSetDevice(device->cudaID));
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
  struct DevGroup {
    typedef std::shared_ptr<DevGroup> SP;

    static SP create(const std::vector<Device::SP> &devices)
    { return std::make_shared<DevGroup>(devices); }
    
    DevGroup(const std::vector<Device::SP> &devices);
    ~DevGroup()
    { printf("MORI DEVICE *GROUP* IS DYING\n"); }
    
    int size() const { return devices.size(); }
    
    std::mutex            mutex;
    std::vector<Device::SP> devices;
  };
  
}
