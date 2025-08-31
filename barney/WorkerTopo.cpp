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

#include "barney/WorkerTopo.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  WorkerTopo::WorkerTopo(const std::vector<Device> &devices,
                         int myOffset, int myCount)
    : allDevices(devices),
      islandOf(devices.size()),
      islandRankOf(devices.size()),
      physicalHostIndexOf(devices.size()),
      physicalGpuIndexOf(devices.size()),
      rankOnHost(devices.size()),
      myOffset(myOffset),
      myCount(myCount),
      _worldRank(devices[myOffset].worldRank)
  {
    for (int gid=0;gid<(int)allDevices.size();gid++) {
      WorkerTopo::Device &dev = allDevices[gid];
      dev.gid = gid;
    }
    numWorkerDevices = 0;
    std::map<int,int> useCountOfDG;
    std::map<size_t,int> nextProcessOnHost;
    std::map<size_t,int> physicalHostIndexOfHostHash;
    std::map<size_t,std::map<size_t,int>> knownGpusInHost;
    
    for (int gid=0;gid<(int)devices.size();gid++) {
      Device dev = devices[gid];
      assert(dev.dataRank >= -1);
      if (dev.dataRank == -1) {
        islandOf[gid] = -1;
        islandRankOf[gid] = -1;
      } else {
        numWorkerDevices++;
        int islandID = useCountOfDG[dev.dataRank]++;
        islandOf[gid] = islandID;
        if (islands.size() <= islandID) islands.resize(islandID+1);
        islandRankOf[gid] = (int)islands[islandID].size();
        islands[islandID].push_back(gid);
      }
      rankOnHost[gid] = nextProcessOnHost[dev.hostNameHash]++;
      // physicalDeviceIndexOf[gid] = nextPhysialGPUInHostHash[dev.hostNameHash]++;
      if (physicalHostIndexOfHostHash.find(dev.hostNameHash) ==
          physicalHostIndexOfHostHash.end())
        physicalHostIndexOfHostHash[dev.hostNameHash]
          = physicalHostIndexOfHostHash.size();
      physicalHostIndexOf[gid]
        = physicalHostIndexOfHostHash[dev.hostNameHash];
      
      if (knownGpusInHost[dev.hostNameHash].find(dev.physicalDeviceHash) == 
          knownGpusInHost[dev.hostNameHash].end()) {
        int newID = (int)knownGpusInHost[dev.hostNameHash].size();
        knownGpusInHost[dev.hostNameHash][dev.physicalDeviceHash] = newID;
      }
      physicalGpuIndexOf[gid]
        = knownGpusInHost[dev.hostNameHash][dev.physicalDeviceHash];
    }
    // some final sanity checks ...
    assert(islands.size() > 0);
    for (auto &island : islands) assert(island.size() == islands[0].size());

    std::string tag = "#bn.topo("+std::to_string(_worldRank)+")";
    if (FromEnv::get()->logTopo) {
      std::stringstream ss; 

      ss << tag << "computed topology as follows:" << std::endl;
      ss << tag << "num devices total " << devices.size() << std::endl;
      ss << tag << "num islands " << islands.size()
         << " island size " << islandSize() << std::endl;
      for (int gid=0;gid<(int)devices.size();gid++) {
        ss << toString(gid,tag);
        ss << "\n";
      }
      std::cout << ss.str();
    }
    
    if (anyGpuIsOverSubscribed())
      std::cout << "#bn: WARNING - at least one physical GPU is over-subcribed!" << std::endl;
  }

  bool WorkerTopo::isDataParallel() const
  {
    return islandSize() != 1;
  }
  
  std::string WorkerTopo::toString(int gid, const std::string &tag) const
  {
    assert(gid >= 0 && gid < allDevices.size());
    std::stringstream ss;
    const auto &dev = allDevices[gid];
    ss << tag << "gid=" << gid << "("
       << dev.worldRank
       << "."
       << dev.local
       << "):";
    // ss << " worker=" << dev.worker;
    // ss << " worldRank=" << dev.worldRank;
    // ss << " local=" << dev.local;
    ss << " dataColor=" << dev.dataRank;
    ss << " island=" << islandOf[gid] << "(rank " << islandRankOf[gid] << ")";
    ss << " hostIdx=" << physicalHostIndexOf[gid];
    ss << " gpuIdx=" << physicalGpuIndexOf[gid];
    // ss << " host:gpuOfs=" << physicalHostIndexOf[gid] << ":" << physicalDeviceIndexOf[gid];
    ss << " phys=" << (int*)dev.hostNameHash << ":" << (int*)dev.physicalDeviceHash;
    return ss.str();
  }
  
  bool WorkerTopo::anyGpuIsOverSubscribed() const
  {
    std::map<std::pair<size_t,size_t>,int> useCount;
    for (auto device : allDevices)
      useCount[{device.hostNameHash,device.physicalDeviceHash}]++;

    for (auto uc : useCount)
      if (uc.second != 1) return true;
    return false;
  }
  
  int WorkerTopo::islandSize() const
  { return (int)islands[0].size(); }
  
  /*! finds ID of device that lived on diven worker:local */
  int WorkerTopo::find(int worker, int local)
  {
    for (int i=0;i<allDevices.size();i++) {
      if (allDevices[i].worker == worker &&
          allDevices[i].local == local)
        return i;
    }
    throw std::runtime_error("could not find given device!?");
  }

}

