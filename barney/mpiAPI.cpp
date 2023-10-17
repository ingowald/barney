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

#include "barney/MPIContext.h"
#include "barney/LocalContext.h"

namespace barney {


  BN_API
  void  bnMPIQueryHardware(BNHardwareInfo *_hardware, MPI_Comm _comm)
  {
    assert(_hardware);
    BNHardwareInfo &hardware = *_hardware;

    assert(_comm != MPI_COMM_NULL);
    mpi::Comm comm(_comm);

    hardware.numRanks = comm.size;
    char hostName[MPI_MAX_PROCESSOR_NAME];
    memset(hostName,0,MPI_MAX_PROCESSOR_NAME);
    int hostNameLen = 0;
    BN_MPI_CALL(Get_processor_name(hostName,&hostNameLen));
    
    std::vector<char> recvBuf(MPI_MAX_PROCESSOR_NAME*comm.size);
    memset(recvBuf.data(),0,recvBuf.size());

    // ------------------------------------------------------------------
    // determine which (world) rank lived on which host, and assign
    // GPUSs
    // ------------------------------------------------------------------
    BN_MPI_CALL(Allgather(hostName,
                          MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
                          recvBuf.data(),
                          /* PER rank size */MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
                          comm.comm));
    std::vector<std::string>  hostNames;
    std::map<std::string,int> ranksOnHost;
    for (int i=0;i<comm.size;i++)  {
      std::string host_i = recvBuf.data()+i*MPI_MAX_PROCESSOR_NAME;
      hostNames.push_back(host_i);
      ranksOnHost[host_i] ++;
    }
    
    hardware.numRanksThisHost = ranksOnHost[hostName];
    hardware.numHosts         = ranksOnHost.size();
    
    // ------------------------------------------------------------------
    // count how many other ranks are already on this same node
    // ------------------------------------------------------------------
    BN_MPI_CALL(Barrier(comm.comm));
    int localRank = 0;
    for (int i=0;i<comm.size;i++) 
      if (hostNames[i] == hostName)
        localRank++;
    BN_MPI_CALL(Barrier(comm.comm));
    hardware.localRank = localRank;
    hardware.numRanksThisHost = ranksOnHost[hostName];

    // ------------------------------------------------------------------
    // assign a GPU to this rank
    // ------------------------------------------------------------------
    int numGPUsOnThisHost;
    cudaGetDeviceCount(&numGPUsOnThisHost);
    if (numGPUsOnThisHost == 0)
      throw std::runtime_error("no GPU on this rank!");
    hardware.numGPUsThisHost = numGPUsOnThisHost;
    hardware.numGPUsThisRank
      = comm.allReduceMin(std::max(hardware.numGPUsThisHost/
                                   hardware.numRanksThisHost,
                                   1));
    assert(hardware.numGPUsThisRank > 0);
  }

  BN_API
  BNContext bnMPIContextCreate(MPI_Comm _comm,
                               /*! which data group(s) this rank will
                                 owl - default is 1 group, with data
                                 group equal to mpi rank */
                               const int *dataGroupsOnThisRank,
                               int  numDataGroupsOnThisRank,
                               /*! which gpu(s) to use for this
                                 process. default is to distribute
                                 node's GPUs equally over all ranks on
                                 that given node */
                               const int *_gpuIDs,
                               int  numGPUs
                               )
  {
    // ------------------------------------------------------------------
    // create vector of data groups; if actual specified by user we
    // use those; otherwise we use IDs
    // [0,1,...numDataGroupsOnThisHost)
    // ------------------------------------------------------------------
    assert(/* data groups == 0 is allowed for passive nodes*/
           numDataGroupsOnThisRank >= 0);
    std::vector<int> dataGroupIDs;
    for (int i=0;i<numDataGroupsOnThisRank;i++)
      dataGroupIDs.push_back
        (dataGroupsOnThisRank
         ? dataGroupsOnThisRank[i]
         : i);

    // ------------------------------------------------------------------
    // create list of GPUs to use for this rank. if specified by user
    // we use this; otherwise we use GPUs in order, split into groups
    // according to how many ranks there are on this host. Ie, if host
    // has four GPUs the first rank will take 0 and 1; and the second
    // one will take 2 and 3.
    // ------------------------------------------------------------------
    BNHardwareInfo hardware;
    bnMPIQueryHardware(&hardware,_comm);
    
    std::vector<int> gpuIDs;
    if (_gpuIDs) {
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back(_gpuIDs[i]);
    } else {
      if (numGPUs < 1)
        numGPUs = hardware.numGPUsThisRank;
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back((hardware.localRank*hardware.numGPUsThisRank
                          + i) % hardware.numGPUsThisHost);
    }
    
    mpi::Comm world(_comm);

    if (world.size == 1) {
      std::cout << "#bn: MPIContextInit, but only one rank - using local context" << std::endl;
      return (BNContext)new LocalContext(dataGroupIDs,
                                         gpuIDs);
    } else {
      bool isActiveWorker = !dataGroupIDs.empty();
      mpi::Comm workers = world.split(isActiveWorker);
      
      return (BNContext)new MPIContext(world,
                                       workers,
                                       isActiveWorker,
                                       dataGroupIDs,
                                       gpuIDs);
    }
  }
  
}
