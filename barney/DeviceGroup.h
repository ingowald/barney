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

  struct TiledFB;
  struct RayQueue;
  
  typedef rtc::GeomType (*GeomTypeCreationFct)(rtc::Device *);
                         
  struct GeomTypeRegistry {
    GeomTypeRegistry(rtc::Device *device);
    rtc::GeomType *get(const std::string &name,
                       GeomTypeCreationFct callback);
    std::map<std::string,rtc::GeomType *> geomTypes;

    rtc::Device *const device;    
    // template<typename CreateGTLambda>
    // // OWLGeomType
    // rtc::GeomType *
    // getOrCreateGeomTypeFor(const std::string &geomTypeString,
    //                        const CreateGTLambda &createGT)
    // {
    //   std::lock_guard<std::mutex> lock(this->mutex);
    //   rtc::GeomType *gt = geomTypes[geomTypeString];
    //   if (gt)
    //     return gt;
      
    //   gt = geomTypes[geomTypeString] = createGT(this);
    //   programsDirty = true;
    //   return gt;
    // }
  };

  
  struct Device {
    typedef std::shared_ptr<Device> SP;
    
    Device(rtc::Device *rtc,
           int contextRank,
           int contextSize,
           int globalIndex,
           int globalIndexStep);
    
    /*! rank and size in the *LOCAL NODE*'s context; ie, these are NOT
        physical Device IDs (a context can use a subset of gpus, as
        well as oversubscribe some!); and they are *not* the 'global'
        device IDs that MPI-wide ray queue cycling would argue about,
        either */
    int                const contextRank;
    int                const contextSize;
    int                const globalIndex;
    int                const globalIndexStep;
    
    void sync() { rtc->sync(); }
    
    /* for ray queue cycling - who to cycle with */
    struct {
      int sendWorkerRank  = -1;
      int sendWorkerLocal = -1;
      int recvWorkerRank  = -1;
      int recvWorkerLocal = -1;
    } rqs;

    int  setActive() const { return rtc->setActive(); }
    void restoreActive(int old) const  { rtc->restoreActive(old); }
    void syncPipelineAndSBT();
    
    bool programsDirty = true;
    bool sbtDirty = true;
    
    GeomTypeRegistry geomTypes;
    rtc::Device *const rtc;
    rtc::Compute *generateRays = 0;
    rtc::Compute *shadeRays = 0;
    rtc::Compute *setTileCoords = 0;
    rtc::Compute *compressTiles = 0;
    rtc::Trace   *traceRays = 0;
    RayQueue     *rayQueue = 0;
  };
  
  /*! stolen from owl/Device: helper class that will set the
    active cuda device (to the device associated with a given
    Context::DeviceData) for the duration fo the lifetime of this
    class, and resets it to whatever it was after class dies */
  struct SetActiveGPU {
    inline SetActiveGPU(const Device *device)
      : savedDevice(device)
    { assert(device); savedActiveDeviceID = device->setActive(); }
    
    inline SetActiveGPU(const Device::SP &device)
      : savedDevice(device.get())
    { savedActiveDeviceID = device->setActive(); }

    inline ~SetActiveGPU()
    { savedDevice->restoreActive(savedActiveDeviceID); }
  private:
    int savedActiveDeviceID = -1;
    const Device *const savedDevice;
  };

  /*! a group of devices that need to share in "something".
    in practive, this is either one of:

    a) the list of devices in a given local model slot

    b) a list of all devices in the local context; or

    c) a single device (eg, the one that does final frame buffer
    assembly and/or denoisign

    In the first case the lmsIdx is the local index of that model slot,
    in the other cases it is '-1'
  */
  struct DevGroup : public std::vector<Device*> {
    typedef std::shared_ptr<DevGroup> SP;
    
    DevGroup(const std::vector<Device*> &devices,
             int numLogical);
    
    /*! the model slot that this
      int const lmsIndx;
      
      /*! *TOTAL* number of logical devices in the context;
      *NOT* how many devices there are in this group. */
    int const numLogical;
  };
  
 //  // still need this?
 //  struct DevGroup {
 //    typedef std::shared_ptr<DevGroup> SP;

 //    DevGroup(int lmsIdx,
 //             const std::vector<int> &contextRanks,
 //             int contextSize,
 //             const std::vector<int> &gpuIDs,
 //             int globalIndex,
 //             int globalIndexStep);
 //    ~DevGroup();
    
 //    int size() const { return (int)devices.size(); }
    


 //    static bool logging() { return false; }
    
 //    void update();
    
 //    // OWLContext               owl = 0;
 //    // OWLRayGen                rg = 0;
 //    // OWLLaunchParams          lp = 0;
 //    std::vector<Device::SP>  devices;
 //    bool programsDirty = true;
 //    bool sbtDirty = true;
 //    /*! local model slot index. this this is the *local index* of the
 //      slot, not the global rank of the the model part that is loaded
 //      into it; i.e., this always starts with '0' on each rank, no
 //      matter what data the app loads into it */
 //    int const lmsIdx;
 //    rtc::DevGroup *rtc = 0;
 // };
  
}
