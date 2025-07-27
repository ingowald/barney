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

namespace BARNEY_NS {
  
  struct WorkerTopo {
    typedef std::shared_ptr<WorkerTopo> SP;
    struct Device {
      /*! the global linear ID of this device; this is the same as its
          index in the 'allDevices[]' vector. it'll be filled in by
          WorkerTopo constructor, so does not have to be set in the
          vector used to construct that WorkerTopo */
      int gid;
      
      /* the worker rank that this device lives on - '0' if local
         rendering, and mpi rank in 'workers' mpi goup if mpi */
      int worker;
      /*! the _global_ world-commm rank of this device */
      int worldRank;
      /*! the local device index for the worker that this device is
        on */
      int local;
      /*! the data rank that this gpu holds */
      int dataRank;

      /*! a hash computed from the host that this GPU is in, should be
          unique across different hosts, but two GPUs in the same host
          should have the same hash */
      size_t hostNameHash;

      /*! hash computed for the PCI busID etc to produce a unique hash
          for each GPU within a host. note this _SHOULD_ be
          independent of CUDA gpu ID ordering, ie, independent of
          whatever CUDA_VISIBLE_DEVIES is set to. As such, two devices
          with same hostnamehash and same physicalDeviceHash WILL mean
          that that gpu is oversubscribed */
      size_t physicalDeviceHash;
    };
    WorkerTopo(const std::vector<Device> &devices,
               int myOffset, int myCount);

    std::string toString(int gid, const std::string &tag="") const;
    bool anyGpuIsOverSubscribed() const;
    
    /*! num GPUs per island */
    int islandSize() const;
    
    /*! finds ID of device that lived on diven worker:local */
    int find(int worker, int local);
    const Device *getLocal(int local) { return &allDevices[myOffset+local]; }
                           
    std::vector<Device> allDevices;
    std::vector<std::vector<int>> islands;
    
    /*! gives, for each logical device, the island it is in */
    std::vector<int> islandOf;

    /*! gives, for each device, an index of the physical host that
        this device is plugged into. Eg, if we ran an mpi rank on two
        machines with 8 gpus each, we will end up with 16 devices, 8
        of which will have physicalHostIndex==0, and 8 will have
        physicalHostIndex==1 */
    std::vector<int> physicalHostIndexOf;
    
    /*! gives, for each device, a *physical* device index relative to
      the host that device is plugged in. Eg, if we run 2 nodes with 8
      GPUs each, then devices 0 and 8 will have phsycal GPU index 0
      (because they're each the respectively first phsyical device in
      the host that they're in), 1 and 9 will have phsycail gpu index
      1, etc.  The physical device index is based on *physical*
      properties like PCI bus ID etc, NOT based on the cuda device
      numbering (and thus, shold be totally independent of slum
      setting CUDA_VISIBLE_DEVICES */
    // std::vector<int> physicalDeviceIndexOf;

    /*! gives, for each logical device, the how many'eth device in its
      island it is */
    std::vector<int> islandRankOf;
    
    
    int worldRank() const { return _worldRank; }
    
    int const myOffset;
    int const myCount;
    int numWorkerDevices;
    int const _worldRank;
  };
  
}
