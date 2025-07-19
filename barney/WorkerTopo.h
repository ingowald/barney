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
    };
    WorkerTopo(const std::vector<Device> &devices,
               int myOffset, int myCount);

    /*! num GPUs per island */
    int islandSize() const;
    
    /*! finds ID of device that lived on diven worker:local */
    int find(int worker, int local);
    const Device *getLocal(int local) { return &allDevices[myOffset+local]; }
                           
    std::vector<Device> allDevices;
    std::vector<std::vector<int>> islands;
    
    /*! gives, for each logical device, the island it is in */
    std::vector<int> islandOf;

    int worldRank() const { return _worldRank; }
    
    /*! gives, for each logical device, the how many'eth device in its
      island it is */
    std::vector<int> islandRankOf;
    int const myOffset;
    int const myCount;
    int numWorkerDevices;
    int const _worldRank;
  };
  
}
