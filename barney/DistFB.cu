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

#include "barney/DistFB.h"
#include "barney/MPIContext.h"

namespace barney {

  DistFB::DistFB(MPIContext *context,
           int owningRank)
      : FrameBuffer(context,owningRank == context->world.rank),
        context(context),
        owningRank(owningRank),
        isOwner(context->world.rank == owningRank),
        ownerIsWorker(context->workerRankOfWorldRank[context->world.rank] != -1)
  {
    if (isOwner) {
      ownerGather.numGPUs = context->numWorkers * context->gpusPerWorker;
      ownerGather.numTilesOnGPU.resize(ownerGather.numGPUs);
    }
    else {
      ownerGather.numGPUs = context->numWorkers * context->gpusPerWorker;
      ownerGather.numTilesOnGPU.resize(0);
    }
  }

  void DistFB::resize(vec2i size, uint32_t *hostFB, float *hostDepth)
  {
    double t0 = getCurrentTime();
    
    FrameBuffer::resize(size, hostFB, hostDepth);
    std::vector<int> tilesOnGPU(perDev.size());
    for (int localID = 0;localID < perDev.size(); localID++) {
      tilesOnGPU[localID] = perDev[localID]->numActiveTiles;
    }

    std::vector<MPI_Request> recv_requests(ownerGather.numGPUs);
    std::vector<MPI_Request> send_requests(tilesOnGPU.size());
    
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu tile count
    // ------------------------------------------------------------------
    if (isOwner) {
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;

        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            &ownerGather.numTilesOnGPU[ggID],1,
                            recv_requests[ggID]);
      }
    }

    if (context->isActiveWorker) {
      for (int localID=0;localID<tilesOnGPU.size();localID++) {
        context->world.send(owningRank,localID,
                            &tilesOnGPU[localID],1,
                            send_requests[localID]);
      }
    }    

    // ------------------------------------------------------------------
    // and wait for those to complete
    // ------------------------------------------------------------------
    if (isOwner)
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
        context->world.wait(recv_requests[ggID]);
    
    if (context->isActiveWorker)
      for (int localID=0;localID<tilesOnGPU.size();localID++)
        context->world.wait(send_requests[localID]);    

    // ------------------------------------------------------------------
    // ------------------------------------------------------------------

    if (isOwner) {
      ownerGather.firstTileOnGPU.resize(ownerGather.numGPUs);
      int sumTiles = 0;
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        ownerGather.firstTileOnGPU[ggID] = sumTiles;
        sumTiles += ownerGather.numTilesOnGPU[ggID];
      }
      ownerGather.numActiveTiles = sumTiles;

      if (ownerGather.finalTiles)
        BARNEY_CUDA_CALL(Free(ownerGather.finalTiles));
      BARNEY_CUDA_CALL(Malloc(&ownerGather.finalTiles,
                              sumTiles*sizeof(*ownerGather.finalTiles)));
      if (ownerGather.tileDescs)
        BARNEY_CUDA_CALL(Free(ownerGather.tileDescs));
      BARNEY_CUDA_CALL(Malloc(&ownerGather.tileDescs,
                              sumTiles*sizeof(*ownerGather.tileDescs)));
      BARNEY_CUDA_SYNC_CHECK();
    }
    
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner) 
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            ownerGather.tileDescs+ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
      }

    if (context->isActiveWorker)
      for (int localID=0;localID<perDev.size();localID++)
        context->world.send(owningRank,localID,
                            perDev[localID]->tileDescs,tilesOnGPU[localID],
                            send_requests[localID]);

    // ------------------------------------------------------------------
    // and wait for those to complete
    // ------------------------------------------------------------------
    if (isOwner)
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
        context->world.wait(recv_requests[ggID]);
    
    if (context->isActiveWorker)
      for (int localID=0;localID<tilesOnGPU.size();localID++)
        context->world.wait(send_requests[localID]);
  }

  void DistFB::ownerGatherFinalTiles()
  {
    std::vector<MPI_Request> recv_requests(ownerGather.numGPUs);
    std::vector<MPI_Request> send_requests(perDev.size());
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner) 
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            ownerGather.finalTiles+ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
      }

    if (context->isActiveWorker)
      for (int localID=0;localID<perDev.size();localID++)
        context->world.send(owningRank,localID,
                            perDev[localID]->finalTiles,
                            perDev[localID]->numActiveTiles,
                            send_requests[localID]);

    // ------------------------------------------------------------------
    // and wait for those to complete
    // ------------------------------------------------------------------
    if (isOwner)
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
        context->world.wait(recv_requests[ggID]);
    
    if (context->isActiveWorker)
      for (int localID=0;localID<perDev.size();localID++)
        context->world.wait(send_requests[localID]);
  }
  
}
