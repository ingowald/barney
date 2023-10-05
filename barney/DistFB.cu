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

namespace barney {

  void DistFB::resize(vec2i size)
  {
    FrameBuffer::resize(size);

    std::vector<int> tilesOnGPU(perGPU.size());
    for (int localID = 0;localID < perGPU.size(); localID++) {
      tilesOnGPU[localID] = perGPU[localID]->numActiveTiles;
    }
    if (comm.rank == 0) {
      // ==================================================================
      // master
      // ==================================================================
      masterGather.numGPUs = comm.size * context->gpuIDs.size();
      masterGather.numTilesOnGPU.resize(masterGather.numGPUs);

      comm.masterGather(// where we'll receive into:
                        masterGather.numTilesOnGPU.data(),
                        // what we're sending:
                        tilesOnGPU.data(),tilesOnGPU.size());

      masterGather.firstTileOnGPU.resize(masterGather.numGPUs);
      int sumTiles = 0;
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        masterGather.firstTileOnGPU[ggID] = sumTiles;
        sumTiles += masterGather.numTilesOnGPU[ggID];
      }
      masterGather.numActiveTiles = sumTiles;

      if (masterGather.finalTiles)
        MORI_CUDA_CALL(Free(masterGather.finalTiles));
      MORI_CUDA_CALL(Malloc(&masterGather.finalTiles,
                            sumTiles*sizeof(*masterGather.finalTiles)));
      if (masterGather.tileDescs)
        MORI_CUDA_CALL(Free(masterGather.tileDescs));
      MORI_CUDA_CALL(MallocManaged(&masterGather.tileDescs,
                            sumTiles*sizeof(*masterGather.tileDescs)));
      MORI_CUDA_SYNC_CHECK();
      
      std::vector<MPI_Request> requests(masterGather.numGPUs);
      int gpusPerRank = context->gpuIDs.size();
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        if (rankOfGPU == 0) {
          auto &dev = context->deviceContexts[localID];
          SetActiveGPU forDuration(dev);
          MORI_CUDA_CALL
            (MemcpyAsync(masterGather.tileDescs+masterGather.firstTileOnGPU[ggID],
                         perGPU[localID]->tileDescs,
                         masterGather.numTilesOnGPU[ggID]*sizeof(TileDesc),
                         cudaMemcpyDefault,
                         dev->stream));
        } else {
          comm.recv(rankOfGPU,localID,
                    masterGather.tileDescs+masterGather.firstTileOnGPU[ggID],
                    masterGather.numTilesOnGPU[ggID],
                    requests[ggID]);
        }
      }
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        if (rankOfGPU == 0) {
          auto &dev = context->deviceContexts[localID];
          MORI_CUDA_CALL(StreamSynchronize(dev->stream));
        } else {
          comm.wait(requests[ggID]);
        }
      }
    } else {
      // ==================================================================
      // worker
      // ==================================================================
      comm.masterGather(// what we're sending:
                        tilesOnGPU.data(),tilesOnGPU.size());
      std::vector<MPI_Request> requests(context->gpuIDs.size());
      for (int localID=0;localID<context->gpuIDs.size();localID++) {
        comm.send(/*to*/0,/*tag*/localID,
                  perGPU[localID]->tileDescs,
                  perGPU[localID]->numActiveTiles,
                  requests[localID]);
      }
      for (int localID=0;localID<context->gpuIDs.size();localID++)
        comm.wait(requests[localID]);
    }
  }

  void DistFB::masterGatherFinalTiles(mpi::Comm &comm)
  {
    std::vector<int> tilesOnGPU(perGPU.size());
    if (comm.rank == 0) {
      // ==================================================================
      // master
      // ==================================================================
      std::vector<MPI_Request> requests(masterGather.numGPUs);
      int gpusPerRank = context->gpuIDs.size();
      // ------------------------------------------------------------------
      // trigger the async copies
      // ------------------------------------------------------------------
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        if (rankOfGPU == 0) {
          auto &dev = context->deviceContexts[localID];
          SetActiveGPU forDuration(dev);
          MORI_CUDA_CALL
            (MemcpyAsync(masterGather.finalTiles+masterGather.firstTileOnGPU[ggID],
                         perGPU[localID]->finalTiles,
                         masterGather.numTilesOnGPU[ggID]*sizeof(*perGPU[localID]->finalTiles),
                         cudaMemcpyDefault,
                         dev->stream));
        } else {
          comm.recv(rankOfGPU,localID,
                    masterGather.finalTiles+masterGather.firstTileOnGPU[ggID],
                    masterGather.numTilesOnGPU[ggID],
                    requests[ggID]);
        }
      }
      // ------------------------------------------------------------------
      // ... and wait for them
      // ------------------------------------------------------------------
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        if (rankOfGPU == 0) {
          auto &dev = context->deviceContexts[localID];
          MORI_CUDA_CALL(StreamSynchronize(dev->stream));
        } else {
          comm.wait(requests[ggID]);
        }
      }
    } else {
      // ==================================================================
      // worker
      // ==================================================================
      std::vector<MPI_Request> requests(context->gpuIDs.size());
      
      // ------------------------------------------------------------------
      // trigger the async copies
      // ------------------------------------------------------------------
      for (int localID=0;localID<context->gpuIDs.size();localID++) {
        comm.send(/*to*/0,/*tag*/localID,
                  perGPU[localID]->finalTiles,
                  perGPU[localID]->numActiveTiles,
                  requests[localID]);
      }
      // ------------------------------------------------------------------
      // ... and wait for them
      // ------------------------------------------------------------------
      for (int localID=0;localID<context->gpuIDs.size();localID++) {
        comm.wait(requests[localID]);
      }
    }
  }
  
}
