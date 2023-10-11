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
      world(comm),
      workers(world.split(isActiveWorker))
  {
    world.assertValid();
    workers.assertValid();

    workerRankOfWorldRank.resize(world.size);
    world.allGather(workerRankOfWorldRank.data(),
                    isActiveWorker?workers.rank:-1);
    worldRankOfWorker.resize(workers.size);
    numWorkers = 0;
    for (int i=0;i<workerRankOfWorldRank.size();i++)
      if (workerRankOfWorldRank[i] != -1) {
        worldRankOfWorker[workerRankOfWorldRank[i]] = i;
        numWorkers++;
      }
    workers.size = numWorkers;
    
    if (isActiveWorker)
      for (int localID=0;localID<gpuIDs.size();localID++) {
        DeviceContext *devCon = perGPU[localID];
        devCon->tileIndexOffset += workers.rank * devCon->tileIndexScale;
        devCon->tileIndexScale  *= workers.size;
      }
    
    /* compute num data groups. this code assumes that user uses IDs
       0,1,2, ...; if thi sis not the case this code will break */
    int myMaxDataID = 0;
    for (auto dataID : dataGroupIDs) {
      assert(dataID >= 0);
      myMaxDataID = std::max(myMaxDataID,dataID);
    }
    
    int numDifferentDataGroups = world.allReduceMax(myMaxDataID)+1;
    assert(!isActiveWorker || (numDifferentDataGroups % dataGroupIDs.size() == 0));

    int myWorkerGPUs
      = isActiveWorker
      ? (int)gpuIDs.size()
      : 0;
    int maxWorkerGPUs = world.allReduceMax(myWorkerGPUs);
    if (isActiveWorker && gpuIDs.size() != maxWorkerGPUs)
      throw std::runtime_error("inconsistent number of GPUs across different workers...");
    gpusPerWorker = maxWorkerGPUs;
    // int numRanksPerIsland = numDifferentDataGroups / (int)dataGroupIDs.size();
    // int numIslands = comm.size / numRanksPerIsland;
  }

  /*! create a frame buffer object suitable to this context */
  FrameBuffer *MPIContext::createFB(int owningRank) 
  {
    return initReference(DistFB::create(this,owningRank));
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int MPIContext::numRaysActiveGlobally() 
  {
    assert(isActiveWorker);
    return workers.allReduceAdd(numRaysActiveLocally());
  }
    
  
  void MPIContext::render(Model *model,
                          const mori::Camera *camera,
                          FrameBuffer *_fb)
  {
    std::cout << "====================== MPIContext::render()" << std::endl;
    DistFB *fb = (DistFB *)_fb;
    if (isActiveWorker) {
      assert(camera);
      renderTiles(model,*camera,fb);
      finalizeTiles(fb);
    }
    // ------------------------------------------------------------------
    // done rendering, now gather all final tiles at master 
    // ------------------------------------------------------------------
    fb->ownerGatherFinalTiles();

    if (fb->isOwner) {
      // ==================================================================
      // now MASTER (who has gathered all the ranks' final tiles) -
      // writes them into proper row-major frame buffer order
      // (writeFinalPixels), then copies them to app FB). only master
      // can/shuld do this - ranks don't even have a 'finalFB' to
      // write into.
      // ==================================================================
      // use default gpu for this:
      assert(fb->perGPU.size() > 0);
      mori::TiledFB::writeFinalPixels(fb->perGPU[0]->device,
                                      fb->finalFB,
                                      fb->numPixels,
                                      fb->ownerGather.finalTiles,
                                      fb->ownerGather.tileDescs,
                                      fb->ownerGather.numActiveTiles);
      // copy to app framebuffer - only if we're the one having that
      // frame buffer of course
      MORI_CUDA_SYNC_CHECK();
      if (fb->hostFB && fb->finalFB != fb->hostFB) {
        MORI_CUDA_CALL(Memcpy(fb->hostFB,fb->finalFB,
                              fb->numPixels.x*fb->numPixels.y*sizeof(uint32_t),
                              cudaMemcpyDefault));
      }
    }
    MORI_CUDA_SYNC_CHECK();
  }
  
}
