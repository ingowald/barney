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

    PING; PRINT(comm.rank); fflush(0);
    
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

      PING; fflush(0);
      comm.barrier();
      PING; fflush(0);
      
      masterGather.firstTileOnGPU.resize(masterGather.numGPUs);
      int sumTiles = 0;
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        masterGather.firstTileOnGPU[ggID] = sumTiles;
        std::cout << "num tiles on " << ggID << " : " << masterGather.numTilesOnGPU[ggID] << std::endl;
        sumTiles += masterGather.numTilesOnGPU[ggID];
      }
      masterGather.numActiveTiles = sumTiles;

      PING; fflush(0);
      if (masterGather.finalTiles)
        MORI_CUDA_CALL(Free(masterGather.finalTiles));
      MORI_CUDA_CALL(Malloc(&masterGather.finalTiles,
                            sumTiles*sizeof(*masterGather.finalTiles)));
      if (masterGather.tileDescs)
        MORI_CUDA_CALL(Free(masterGather.tileDescs));
      MORI_CUDA_CALL(Malloc(&masterGather.tileDescs,
                            sumTiles*sizeof(*masterGather.tileDescs)));
      std::cout << "allocated master tile offsets " << masterGather.tileDescs << ", size " << sumTiles << std::endl;

      MORI_CUDA_SYNC_CHECK();
      
      PING; fflush(0);
      std::vector<MPI_Request> requests(masterGather.numGPUs);
      int gpusPerRank = context->gpuIDs.size();
      PRINT(gpusPerRank); fflush(0);
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        PING; fflush(0);
        if (rankOfGPU == 0) {
          SetActiveGPU forDuration(context->deviceContexts[localID]);
          PING; fflush(0);
          MORI_CUDA_CALL
            (Memcpy(masterGather.tileDescs+masterGather.firstTileOnGPU[ggID],
                    perGPU[localID]->tileDescs,
                    masterGather.numTilesOnGPU[ggID]*sizeof(TileDesc),
                    cudaMemcpyDefault));
          // MORI_CUDA_CALL
          //   (MemcpyAsync(masterGather.tileOffsets+masterGather.firstTileOnGPU[ggID],
          //                perGPU[localID]->tileOffsets,
          //                masterGather.numTilesOnGPU[ggID]*sizeof(TileDesc),
          //                cudaMemcpyDefault,
          //                context->deviceContexts[localID]->stream));
          PING; fflush(0);
        } else {
          PING; fflush(0);
          PRINT(masterGather.tileDescs); fflush(0);
          PRINT(masterGather.tileDescs+masterGather.firstTileOnGPU[ggID]);
          PRINT(masterGather.numTilesOnGPU[ggID]);
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
          MORI_CUDA_CALL(StreamSynchronize(context->deviceContexts[localID]->stream));
        } else {
          comm.wait(requests[ggID]);
        }
      }
    } else {
      PING; fflush(0);
      // ==================================================================
      // worker
      // ==================================================================
      comm.masterGather(// what we're sending:
                        tilesOnGPU.data(),tilesOnGPU.size());
      PING; fflush(0);
      comm.barrier();
      PING; fflush(0);
      std::vector<MPI_Request> requests(context->gpuIDs.size());
      for (int localID=0;localID<context->gpuIDs.size();localID++) {
        PRINT(perGPU[localID]->tileDescs);
        PING; fflush(0);
        comm.send(/*to*/0,/*tag*/localID,
                  perGPU[localID]->tileDescs,
                  perGPU[localID]->numActiveTiles,
                  requests[localID]);
        PING; fflush(0);
      }
      PING; fflush(0);
      for (int localID=0;localID<context->gpuIDs.size();localID++)
        comm.wait(requests[localID]);
      PING; fflush(0);
    }
    
    PING; PRINT(comm.rank); fflush(0);
  }

  void DistFB::masterGatherFinalTiles(mpi::Comm &comm)
  {
    std::vector<int> tilesOnGPU(perGPU.size());
    // for (int localID = 0;localID < perGPU.size(); localID++) {
    //   tilesOnGPU[localID] = perGPU[localID]->numActiveTiles;
    // }
    if (comm.rank == 0) {
      comm.barrier();
      std::vector<MPI_Request> requests(masterGather.numGPUs);
      int gpusPerRank = context->gpuIDs.size();
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        if (rankOfGPU == 0) {
          SetActiveGPU forDuration(context->deviceContexts[localID]);
          MORI_CUDA_CALL
            (Memcpy(masterGather.finalTiles+masterGather.firstTileOnGPU[ggID],
                    perGPU[localID]->finalTiles,
                    masterGather.numTilesOnGPU[ggID]*sizeof(*perGPU[localID]->finalTiles),
                    cudaMemcpyDefault));
        } else {
          std::cout << "master receiving " << masterGather.numTilesOnGPU[ggID] << " (from rank " << rankOfGPU << ")" << std::endl;
          
          comm.recv(rankOfGPU,localID,
                    masterGather.finalTiles+masterGather.firstTileOnGPU[ggID],
                    masterGather.numTilesOnGPU[ggID],
                    requests[ggID]);
        }
      }
      for (int ggID = 0; ggID < masterGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / gpusPerRank;
        int localID   = ggID % gpusPerRank;
        if (rankOfGPU == 0) {
          PING;
          MORI_CUDA_CALL(StreamSynchronize(context->deviceContexts[localID]->stream));
        } else {
          PING;
          comm.wait(requests[ggID]);
        }
      }
    } else {
      PING; fflush(0);
      // ==================================================================
      // worker
      // ==================================================================
      // comm.masterGather(// what we're sending:
      //                   tilesOnGPU.data(),tilesOnGPU.size());
      PING; fflush(0);
      comm.barrier();
      PING; fflush(0);
      std::vector<MPI_Request> requests(context->gpuIDs.size());
      for (int localID=0;localID<context->gpuIDs.size();localID++) {
        PRINT(perGPU[localID]->finalTiles);
        PING; fflush(0);
        comm.send(/*to*/0,/*tag*/localID,
                  perGPU[localID]->finalTiles,
                  perGPU[localID]->numActiveTiles,
                  requests[localID]);
        PING; fflush(0);
      }
      PING; fflush(0);
      for (int localID=0;localID<context->gpuIDs.size();localID++)
        comm.wait(requests[localID]);
      PING; fflush(0);
    }
    comm.barrier();
    PING; PRINT(comm.rank); fflush(0);
    comm.barrier();
  }
  
}
