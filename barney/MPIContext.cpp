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
#include "barney/DistFB.h"

namespace barney {

  MPIContext::MPIContext(const mpi::Comm &comm,
                         const std::vector<int> &dataGroupIDs,
                         const std::vector<int> &gpuIDs)
    : Context(dataGroupIDs,gpuIDs),
      comm(comm)
  {
    comm.assertValid();

    for (int localID=0;localID<gpuIDs.size();localID++) {
      DeviceContext *devCon = deviceContexts[localID];
      devCon->tileIndexOffset += comm.rank * devCon->tileIndexScale;
      devCon->tileIndexScale  *= comm.size;
    }
    
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
  { return initReference(DistFB::create(this,comm)); }
    // ,
    //                                     comm.rank*gpuIDs.size(),
    //                                     comm.size*gpuIDs.size())); }

  void MPIContext::render(Model *model,
                          const BNCamera *camera,
                          FrameBuffer *fb,
                          uint32_t *appFB)
  {
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      // computation of Tile::accum color
      renderTiles(this,localID,model,fb,camera);
    
    for (int localID = 0; localID < gpuIDs.size(); localID++)
      // accum to rgba conversion:
      fb->perGPU[localID]->finalizeTiles();

    for (int localID = 0; localID < gpuIDs.size(); localID++) {
      auto &devFB = *fb->perGPU[localID];
      SetActiveGPU forDuration(devFB.device);
      mori::TiledFB::writeFinalPixels(fb->finalFB,
                                      fb->numPixels,
                                      devFB.finalTiles,
                                      devFB.tileDescs,
                                      devFB.numActiveTiles,
                                      devFB.device->stream);
    }


    for (int localID = 0; localID < gpuIDs.size(); localID++)
      fb->perGPU[localID]->sync();

    // ------------------------------------------------------------------
    // done rendering, now gather all final tiles at master 
    // ------------------------------------------------------------------
    ((DistFB *)fb)->masterGatherFinalTiles(comm);

    if (appFB) {
      // ==================================================================
      // now MASTER (who has gathered all the ranks' final tiles) -
      // writes them into proper row-major frame buffer order
      // (writeFinalPixels), then copies them to app FB). only master
      // can/shuld do this - ranks don't even have a 'finalFB' to
      // write into.
      // ==================================================================
      // use default gpu for this:
      mori::TiledFB::writeFinalPixels(fb->finalFB,
                                      fb->numPixels,
                                      ((DistFB *)fb)->masterGather.finalTiles,
                                      ((DistFB *)fb)->masterGather.tileDescs,
                                      ((DistFB *)fb)->masterGather.numActiveTiles,
                                      (cudaStream_t)0);
      PING; PRINT(comm.rank); fflush(0);
      MORI_CUDA_SYNC_CHECK();
      // copy to app framebuffer - only if we're the one having that
      // frame buffer of course
      MORI_CUDA_CALL(Memcpy(appFB,fb->finalFB,
                            fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                            cudaMemcpyDefault));
      PING; PRINT(comm.rank); fflush(0);
    }
    comm.barrier();
    PING; PRINT(comm.rank); fflush(0);
    MORI_CUDA_SYNC_CHECK();
  }
  
}
