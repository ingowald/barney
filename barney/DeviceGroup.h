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

#include "barney/common.h"

namespace barney {

  struct DevGroup;
  
  struct Device {
    typedef std::shared_ptr<Device> SP;
    
    Device(DevGroup *devGroup,
           int gpuID,
           int owlID,
           int globalIndex,
           int globalIndexStep);

    int                const cudaID;
    int                const owlID;
    int                const globalIndex;
    int                const globalIndexStep;
    DevGroup          *const devGroup;
    cudaStream_t       const launchStream;

    /* for ray queue cycling - who to cycle with */
    struct {
      int sendWorkerRank  = -1;
      int sendWorkerLocal = -1;
      int recvWorkerRank  = -1;
      int recvWorkerLocal = -1;
    } rqs;
  };
  
  /*! stolen from owl/Device: helper class that will set the
      active cuda device (to the device associated with a given
      Context::DeviceData) for the duration fo the lifetime of this
      class, and resets it to whatever it was after class dies */
  struct SetActiveGPU {
    inline SetActiveGPU(const Device *device)
    {
      // assert(device);
      BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      BARNEY_CUDA_CHECK(cudaSetDevice(device?device->cudaID:0));
    }
    inline SetActiveGPU(const Device::SP &device)
    {
      // assert(device);
      BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      BARNEY_CUDA_CHECK(cudaSetDevice(device?device->cudaID:0));
    }
    
    inline SetActiveGPU(int cudaDeviceID)
    {
      BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      BARNEY_CUDA_CHECK(cudaSetDevice(cudaDeviceID));
    }
    inline ~SetActiveGPU()
    {
      BARNEY_CUDA_CHECK_NOTHROW(cudaSetDevice(savedActiveDeviceID));
    }
  private:
    int savedActiveDeviceID = -1;
  };
  

  // still need this?
  struct DevGroup {
    typedef std::shared_ptr<DevGroup> SP;

    DevGroup(int ldgID,
             const std::vector<int> &gpuIDs,
             int globalIndex,
             int globalIndexStep);
    
    int size() const { return devices.size(); }
    
    OWLGeomType getOrCreateGeomTypeFor(const std::string &geomTypeString,
                                       OWLGeomType (*createOnce)(DevGroup *));
    void update();
    
    std::map<std::string,OWLGeomType> geomTypes;
    std::mutex               mutex;
    OWLContext               owl = 0;
    OWLRayGen                rg = 0;
    OWLLaunchParams          lp = 0;
    std::vector<Device::SP>  devices;
    bool programsDirty = true;
    bool sbtDirty = true;
    /*! local device group ID */
    int const ldgID;

  };
  
}
