// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/fb/DistFB.h"
#include "barney/MPIContext.h"

namespace BARNEY_NS {

  struct UnpackTiles {
    vec2i numPixels;
    rtc::float4 *out_rgba;
    vec3f *normals;
    CompressedTile *tiles;
    TileDesc *descs;
    
    __rtc_device void run(const rtc::ComputeInterface &ci);
  };

#if RTC_DEVICE_CODE
  __rtc_device void UnpackTiles::run(const rtc::ComputeInterface &ci)
  {
    int tileIdx = ci.getBlockIdx().x;
      
    const CompressedTile &tile = tiles[tileIdx];
    const TileDesc        desc = descs[tileIdx];
      
    int subIdx = ci.getThreadIdx().x;
    int iix = subIdx % tileSize;
    int iiy = subIdx / tileSize;
    int ix = desc.lower.x + iix;
    int iy = desc.lower.y + iiy;
    if (ix >= numPixels.x) return;
    if (iy >= numPixels.y) return;
    int idx = ix + numPixels.x*iy;
      
    uint32_t rgba8 = tile.rgba[subIdx];
    vec4f rgba = from_8bit(rgba8);
    float scale = float(tile.scale[subIdx]);
    rgba.x *= scale;
    rgba.y *= scale;
    rgba.z *= scale;
    vec3f normal = tile.normal[subIdx].get3f();
    // float depth = tile.depth[subIdx];

    out_rgba[idx] = (const rtc::float4&)rgba;
    // if (depths)
    //   depths[idx] = depth;
    if (normals)
      normals[idx] = normal;
  }
#endif
  
  void FrameBuffer::unpackTiles()
  {
    UnpackTiles args = {
      numPixels,
      (rtc::float4*)linearColor,
      linearNormal,
      // linearDepth,
      gatheredTilesOnOwner.compressedTiles,
      gatheredTilesOnOwner.tileDescs
    };
    auto device = getDenoiserDevice();
    device->unpackTiles->launch(gatheredTilesOnOwner.numActiveTiles,
                                pixelsPerTile,
                                &args);
    device->sync();
  }

  
  DistFB::DistFB(MPIContext *context,
                 const DevGroup::SP &devices,
                 int owningRank)
    : FrameBuffer(context,
                  devices,
                  owningRank == context->world.rank),
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

  void DistFB::resize(vec2i size,
                      uint32_t channels)
  {
    double t0 = getCurrentTime();
    
    FrameBuffer::resize(size, channels);
    std::vector<int> tilesOnGPU(devices->numLogical);
    for (auto device : *devices) {
      tilesOnGPU[device->contextRank]
        = getPLD(device)->tiledFB->numActiveTiles;
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
      gatheredTilesOnOwner.numActiveTiles = sumTiles;

      Device *frontDev = getDenoiserDevice();
      
      if (gatheredTilesOnOwner.compressedTiles) {
        frontDev->rtc->freeMem(gatheredTilesOnOwner.compressedTiles);
        gatheredTilesOnOwner.compressedTiles = 0;
      }
      
      if (gatheredTilesOnOwner.tileDescs) {
        frontDev->rtc->freeMem(gatheredTilesOnOwner.tileDescs);
        gatheredTilesOnOwner.tileDescs = 0;
      }
      
      gatheredTilesOnOwner.compressedTiles
        = (CompressedTile *)frontDev->rtc->allocMem
        (sumTiles*sizeof(*gatheredTilesOnOwner.compressedTiles));
      
      gatheredTilesOnOwner.tileDescs
        = (TileDesc *)frontDev->rtc->allocMem
        (sumTiles*sizeof(*gatheredTilesOnOwner.tileDescs));
    }
    
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner) 
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            gatheredTilesOnOwner.tileDescs+ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
      }

    if (context->isActiveWorker)
      for (auto device : *devices) 
        context->world.send(owningRank,
                            device->contextRank,
                            getPLD(device)->tiledFB->tileDescs,
                            tilesOnGPU[device->contextRank],
                            send_requests[device->contextRank]);

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

  void DistFB::ownerGatherCompressedTiles()
  {
    std::vector<MPI_Request> recv_requests(ownerGather.numGPUs);
    std::vector<MPI_Request> send_requests(devices->size());
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner) 
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            gatheredTilesOnOwner.compressedTiles+ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
      }

    if (context->isActiveWorker)
      for (auto device : *devices)
      // for (int localID=0;localID<perDev.size();localID++)
        context->world.send(owningRank,device->contextRank,//localID,
                            getPLD(device)->tiledFB->compressedTiles,
                            getPLD(device)->tiledFB->numActiveTiles,
                            send_requests[device->contextRank]);

    // ------------------------------------------------------------------
    // and wait for those to complete
    // ------------------------------------------------------------------
    if (isOwner)
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
        context->world.wait(recv_requests[ggID]);
    
    if (context->isActiveWorker)
      for (auto device : *devices)
        context->world.wait(send_requests[device->contextRank]);
  }
  
  RTC_EXPORT_COMPUTE1D(unpackTiles,UnpackTiles);
}
