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

#include "barney/common/barney-common.h"
#include "rtcore/common/Backend.h"
#include "rtcore/common/RTCore.h"

namespace barney {

  struct DevGroup;
  
  struct Device {
    typedef std::shared_ptr<Device> SP;
    
    Device(DevGroup *devGroup,
           rtc::Device *rtc,
           int contextRank,
           int contextSize,
           // int gpuID,
           // int owlID,
           int globalIndex,
           int globalIndexStep);

    int                const contextRank;
    int                const contextSize;
    // int                const cudaID;
    // int                const owlID;
    int                const globalIndex;
    int                const globalIndexStep;
    DevGroup          *const devGroup;
    // cudaStream_t       const launchStream;

    /* for ray queue cycling - who to cycle with */
    struct {
      int sendWorkerRank  = -1;
      int sendWorkerLocal = -1;
      int recvWorkerRank  = -1;
      int recvWorkerLocal = -1;
    } rqs;

    int setActive() const { return rtc->setActive(); }
    void restoreActive(int old) const  { rtc->restoreActive(old); }

    rtc::Device *const rtc;
  };
  
  /*! stolen from owl/Device: helper class that will set the
    active cuda device (to the device associated with a given
    Context::DeviceData) for the duration fo the lifetime of this
    class, and resets it to whatever it was after class dies */
  struct SetActiveGPU {
    inline SetActiveGPU(const Device *device)
      : savedDevice(device)
    {
      if (!device) return;
      // auto backend = rtc::Backend::get();
      savedActiveDeviceID = device->setActive();
      // rtbackend->getActiveGPU();
      // backend->setActiveGPU(device->rtc->physicalID);
      // assert(device);
      // if (device) {
      //   BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      //   BARNEY_CUDA_CHECK(cudaSetDevice(device?device->cudaID:0));
      // }
    }
    inline SetActiveGPU(const Device::SP &device)
      : savedDevice(device.get())
    {
      savedActiveDeviceID = device->setActive();
      // auto backend = rtc::Backend::get();
      // savedActiveDeviceID = device->setActive();
      // savedActiveDeviceID = backend->getActiveGPU();
      // backend->setActiveGPU(device->rtc->physicalID);
      // assert(device);
      // BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      // BARNEY_CUDA_CHECK(cudaSetDevice(device?device->cudaID:0));
    }

    // inline SetActiveGPU(int physicalID)
    // {
    //   auto backend = rtc::Backend::get();
    //   savedActiveDeviceID = backend->getActiveGPU();
    //   backend->setActiveGPU(physicalID);
    //   // BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
    //   // BARNEY_CUDA_CHECK(cudaSetDevice(cudaDeviceID));
    // }
    inline ~SetActiveGPU()
    {
      savedDevice->restoreActive(savedActiveDeviceID);
      
      // if (savedActiveDeviceID < 0)
      //   return;
      // auto backend = rtc::Backend::get();
      // backend->setActiveGPU(savedActiveDeviceID);
      // if (savedActiveDeviceID >= 0)
      //   BARNEY_CUDA_CHECK_NOTHROW(cudaSetDevice(savedActiveDeviceID));
    }
  private:
    int savedActiveDeviceID = -1;
    const Device *const savedDevice;
  };
  

  // still need this?
  struct DevGroup {
    typedef std::shared_ptr<DevGroup> SP;

    DevGroup(int lmsIdx,
             const std::vector<int> &contextRanks,
             int contextSize,
             const std::vector<int> &gpuIDs,
             int globalIndex,
             int globalIndexStep);
    ~DevGroup();
    
    int size() const { return (int)devices.size(); }
    
    template<typename CreateGTLambda>
    // OWLGeomType
    rtc::GeomType *
    getOrCreateGeomTypeFor(const std::string &geomTypeString,
                           const CreateGTLambda &createGT)
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      rtc::GeomType *gt = geomTypes[geomTypeString];
      if (gt)
        return gt;
      
      gt = geomTypes[geomTypeString] = createGT(this);
      programsDirty = true;
      return gt;
    }


    static bool logging() { return false; }
    
    void update();
    
    std::map<std::string,rtc::GeomType *> geomTypes;
    std::mutex               mutex;
    // OWLContext               owl = 0;
    // OWLRayGen                rg = 0;
    // OWLLaunchParams          lp = 0;
    std::vector<Device::SP>  devices;
    bool programsDirty = true;
    bool sbtDirty = true;
    /*! local model slot index. this this is the *local index* of the
      slot, not the global rank of the the model part that is loaded
      into it; i.e., this always starts with '0' on each rank, no
      matter what data the app loads into it */
    int const lmsIdx;
    rtc::DevGroup *rtc = 0;
    rtc::ComputeKernel *setTileCoordsKernel;
    rtc::ComputeKernel *compressTilesKernel;
    rtc::ComputeKernel *generateRaysKernel;
    rtc::ComputeKernel *shadeRaysKernel;
    rtc::TraceKernel   *traceRaysKernel;
 };
  
}
