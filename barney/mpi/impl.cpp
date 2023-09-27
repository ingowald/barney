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

#include "barney.h"
#include "barney/mpi/MPIWrappers.h"

namespace barney {

  BN_API
  void  bnQueryHardwareMPI(BNHardwareInfo *_hardware, MPI_Comm _comm)
  {
    assert(_hardware);
    BNHardwareInfo &hardware = *_hardware;

    assert(_comm != MPI_COMM_NULL);
    mpi::Comm comm(_comm);

    
  }

  BN_API
  BNContext bnContextCreateMPI(MPI_Comm _comm,
                               /*! which data group(s) this rank will
                                 owl - default is 1 group, with data
                                 group equal to mpi rank */
                               int *dataGroupsOnThisRank,
                               int  numDataGroupsOnThisRank,
                               /*! which gpu(s) to use for this
                                 process. default is to distribute
                                 node's GPUs equally over all ranks on
                                 that given node */
                               int *gpuIDs,
                               int  numGPUs
                               )
  {
    assert(dataGroupsOnThisRank != nullptr);
    assert(numDataGroupsOnThisRank > 0);
    
    mpi::Comm comm(_comm);
    comm.assertValid();
    PING;

    /* compute num data groups. this code assumes that user uses IDs
       0,1,2, ...; if thi sis not the case this code will break */
    int myMaxDataID = 0;
    for (int i=0;i<numDataGroupsOnThisRank;i++) {
      int dataID = dataGroupsOnThisRank[i];
      assert(dataID >= 0);
      myMaxDataID = std::max(myMaxDataID,dataID);
    }
    
    int numDifferentData = comm.allReduceMax(myMaxDataID)+1;
    PRINT(numDifferentData);

    return 0;
  }
  
}
