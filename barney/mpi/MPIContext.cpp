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

#include "barney/mpi/MPIContext.h"
#include "barney/mpi/DistFB.h"

namespace barney {

  MPIContext::MPIContext(const mpi::Comm &comm,
                         const std::vector<int> &dataGroupIDs,
                         const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs),
      comm(comm)
  {
    comm.assertValid();
    
    /* compute num data groups. this code assumes that user uses IDs
       0,1,2, ...; if thi sis not the case this code will break */
    int myMaxDataID = 0;
    for (auto dataID : dataGroupIDs) {
      assert(dataID >= 0);
      myMaxDataID = std::max(myMaxDataID,dataID);
    }
    
    int numDifferentDataGroups = comm.allReduceMax(myMaxDataID)+1;
    PRINT(numDifferentDataGroups);
    assert(numDifferentDataGroups % dataGroupIDs.size() == 0);
    int numRanksPerIsland = numDifferentDataGroups / (int)dataGroupIDs.size();
    int numIslands = comm.size / numRanksPerIsland;
  }

  /*! create a frame buffer object suitable to this context */
  FrameBuffer *MPIContext::createFB() 
  { return initReference(DistFB::create(this,
                                        comm.rank*gpuIDs.size(),
                                        comm.size*gpuIDs.size())); }

  void MPIContext::render(Model *model,
                          const BNCamera *camera,
                          FrameBuffer *fb,
                          uint32_t *appFB)
  {
    DistFB *local = (DistFB *)fb;

    for (int localID = 0; localID < gpuIDs.size(); localID++)
      renderTiles(this,localID,model,fb,camera);
    
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      fb->perGPU[localID]->writeFinal(local->finalFB,perGPU[localID]->stream);

    for (int localID = 0; localID < gpuIDs.size(); localID++)
      MORI_CUDA_CALL(StreamSynchronize(perGPU[localID]->stream));

    throw std::runtime_error("need to gather to master here ...");
    
    MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
                          fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                          cudaMemcpyDefault));
    MORI_CUDA_SYNC_CHECK();
  }
  
}
