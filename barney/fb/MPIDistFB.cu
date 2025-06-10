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

#include "barney/fb/MPIDistFB.h"
#include "barney/MPIContext.h"
#include "rtcore/ComputeInterface.h"
#include "barney/common/math.h"

namespace BARNEY_NS {

#if 0
  /*! read one of the auxiliary (not color or normal) buffers into
    the given (device-writeable) staging area; this will at the
    least incur some reformatting from tiles to linear (if local
    node), possibly some gpu-gpu transfer (local node w/ more than
    one gpu) and possibly some mpi communication (distFB) */
  void MPIDistFB::gatherAuxChannel(BNFrameBufferChannel channel)
  {
    // ------------------------------------------------------------------
    // gather all (packed) tiles from all clients
    // ------------------------------------------------------------------
    std::vector<MPI_Request> recv_requests(isOwner
                                           ?ownerGather.numGPUs
                                           :0);
    std::vector<MPI_Request> send_requests(context->isActiveWorker
                                           ?devices->size()
                                           :0);
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner) {
      AuxChannelTile *aux_recv = 0;
      switch(channel) {
      case BN_FB_DEPTH:
        aux_recv = gatheredTilesOnOwner.auxChannelTiles.depth;
        break;
      case BN_FB_PRIMID:
        aux_recv = gatheredTilesOnOwner.auxChannelTiles.primID;
        break;
      case BN_FB_INSTID:
        aux_recv = gatheredTilesOnOwner.auxChannelTiles.instID;
        break;
      case BN_FB_OBJID:
        aux_recv = gatheredTilesOnOwner.auxChannelTiles.objID;
        break;
      default:
        throw std::runtime_error("unsupported aux channel in sending aux!?");
      };
      /* OWNER: create receive requests for all compressed tiles from
         all ranks. this also include the ones we send from our own
         GPUs, so do NOT wait here */
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            aux_recv+ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
      }
    }
    
    if (context->isActiveWorker) {
      /* worker: compress all local tiles (on all gpus), and trigger
         send requests to transfer those to master */
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto tiledFB = getFor(device);
        auto pld = getPLD(device);
        AuxChannelTile *aux_send = 0;
        switch(channel) {
        case BN_FB_DEPTH:
          aux_send = tiledFB->auxTiles.depth;
          break;
        case BN_FB_PRIMID:
          aux_send = tiledFB->auxTiles.primID;
          break;
        case BN_FB_INSTID:
          aux_send = tiledFB->auxTiles.instID;
          break;
        case BN_FB_OBJID:
          aux_send = tiledFB->auxTiles.objID;
          break;
        default:
          throw std::runtime_error("unsupported aux channel in sending aux!?");
        };
        context->world.send(owningRank,device->contextRank(),
                            aux_send,
                            tiledFB->numActiveTilesThisGPU,
                            send_requests[device->contextRank()]);
      }
    }
    // ------------------------------------------------------------------
    // wait for all send/recv requests to complete - master should
    // then have all tiles from all gpus, in compressed form
    // ------------------------------------------------------------------
    if (isOwner)
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
        context->world.wait(recv_requests[ggID]);
    
    if (context->isActiveWorker)
      for (auto device : *devices)
        context->world.wait(send_requests[device->contextRank()]);
  }

  /*! gather color (and optionally, if not null) linear normal, from
    all GPUs (and ranks). lienarColor and lienarNormal are
    device-writeable 2D linear arrays of numPixel size;
    linearcolor may be null. */
  void MPIDistFB::gatherColorChannel(/*float4 or rgba8*/void *linearColor,
                                  BNDataType gatherType,
                                  vec3f *linearNormal)
  {
    // ------------------------------------------------------------------
    // gather all (packed) tiles from all clients
    // ------------------------------------------------------------------
    std::vector<MPI_Request> recv_requests(isOwner
                                           ?((needNormals?2:1)*ownerGather.numGPUs)
                                           :0);
    std::vector<MPI_Request> send_requests((needNormals?2:1)*devices->size());
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner) {
      /* OWNER: create receive requests for all compressed tiles from
         all ranks. this also include the ones we send from our own
         GPUs, so do NOT wait here */
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            gatheredTilesOnOwner.compressedColorTiles
                            +ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
        if (needNormals)
          context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                              gatheredTilesOnOwner.compressedNormalTiles
                              +ownerGather.firstTileOnGPU[ggID],
                              ownerGather.numTilesOnGPU[ggID],
                              recv_requests[ownerGather.numGPUs+ggID]);
      }
    }
    
    if (context->isActiveWorker) {
      /* worker: compress all local tiles (on all gpus), and trigger
         send requests to transfer those to master */
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto tiledFB = getFor(device);
        auto pld = getPLD(device);
        float accumScale = 1.f/accumID;
        CompressTiles kernel = {
          pld->localSend.compressedColorTiles,
          pld->localSend.compressedNormalTiles,
          tiledFB->accumTiles,
          accumScale
        };
        pld->compressTiles->launch(tiledFB->numActiveTilesThisGPU,pixelsPerTile,&kernel);
      }
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        auto tiledFB = getFor(device);
        auto pld = getPLD(device);
        device->sync();
        context->world.send(owningRank,device->contextRank(),
                            pld->localSend.compressedColorTiles,
                            tiledFB->numActiveTilesThisGPU,
                            send_requests[device->contextRank()]);
        if (needNormals)
          context->world.send(owningRank,device->contextRank(),
                              pld->localSend.compressedNormalTiles,
                              tiledFB->numActiveTilesThisGPU,
                              send_requests[devices->size()+device->contextRank()]);
      }
    }
    // ------------------------------------------------------------------
    // wait for all send/recv requests to complete - master should
    // then have all tiles from all gpus, in compressed form
    // ------------------------------------------------------------------
    if (isOwner)
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
        context->world.wait(recv_requests[ggID]);
    
    if (context->isActiveWorker)
      for (auto device : *devices)
        context->world.wait(send_requests[device->contextRank()]);
    
    // ------------------------------------------------------------------
    // all (packed) tiles received; unpack (on owner only)
    // ------------------------------------------------------------------
    if (isOwner) {
      UnpackTiles args = {
        numPixels,
        linearColor,
        gatherType,
        linearNormal,
        gatheredTilesOnOwner.compressedColorTiles,
        gatheredTilesOnOwner.compressedNormalTiles,
        gatheredTilesOnOwner.tileDescs
      };
      auto device = getDenoiserDevice();
      SetActiveGPU forDuration(device);
      getPLD(device)->unpackTiles->launch(gatheredTilesOnOwner.numActiveTiles,
                                          pixelsPerTile,
                                          &args);
      device->sync();
    }
  }

  void MPIDistFB::resize(BNDataType colorFormat,
                      vec2i size,
                      uint32_t channels)
  {
    freeChannelData();
    FrameBuffer::resize(colorFormat, size, channels);

    // ------------------------------------------------------------------
    /* check if we need to have a normal channel for deonising (on
       owner), and communicate this to all workers */
    // ------------------------------------------------------------------
    if (isOwner) {
      this->needNormals = denoiser != 0;
      context->world.bc_send(&this->needNormals,sizeof(this->needNormals));
    } else {
      context->world.bc_recv(&this->needNormals,sizeof(this->needNormals));
    }
    
    // ------------------------------------------------------------------
    /* allocate compressed tiles mem - one for each tiles in the
       corresponding tiledFB */
    // ------------------------------------------------------------------
    std::vector<int> tilesOnGPU(devices->numLogical);
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      
      TiledFB *tiledFB = getFor(device);
      tilesOnGPU[device->contextRank()]
        = tiledFB->numActiveTilesThisGPU;
      
      auto pld = getPLD(device);
      pld->localSend.compressedColorTiles
        = (CompressedColorTile*)device->rtc->allocMem
        (tiledFB->numActiveTilesThisGPU*sizeof(CompressedColorTile));
      if (needNormals) 
        pld->localSend.compressedNormalTiles
          = (CompressedNormalTile*)device->rtc->allocMem
          (tiledFB->numActiveTilesThisGPU*sizeof(CompressedNormalTile));
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
      
      gatheredTilesOnOwner.compressedColorTiles
        = (CompressedColorTile *)frontDev->rtc->allocMem
        (sumTiles*sizeof(*gatheredTilesOnOwner.compressedColorTiles));

      if (denoiser)
        gatheredTilesOnOwner.compressedNormalTiles
          = (CompressedNormalTile *)frontDev->rtc->allocMem
          (sumTiles*sizeof(*gatheredTilesOnOwner.compressedNormalTiles));

      if (channels & BN_FB_DEPTH)
        gatheredTilesOnOwner.auxChannelTiles.depth
          = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
      if (channels & BN_FB_PRIMID)
        gatheredTilesOnOwner.auxChannelTiles.primID
          = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
      if (channels & BN_FB_INSTID)
        gatheredTilesOnOwner.auxChannelTiles.instID
          = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
      if (channels & BN_FB_OBJID)
        gatheredTilesOnOwner.auxChannelTiles.objID
          = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));

      gatheredTilesOnOwner.tileDescs
        = (TileDesc *)frontDev->rtc->allocMem
        (sumTiles*sizeof(*gatheredTilesOnOwner.tileDescs));
    }
    
    // ------------------------------------------------------------------
    // trigger all sends and receives - for gpu descs
    // ------------------------------------------------------------------
    if (isOwner)  {
      for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
        int rankOfGPU = ggID / context->gpusPerWorker;
        int localID   = ggID % context->gpusPerWorker;
        context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
                            gatheredTilesOnOwner.tileDescs+ownerGather.firstTileOnGPU[ggID],
                            ownerGather.numTilesOnGPU[ggID],
                            recv_requests[ggID]);
      }
    }

    if (context->isActiveWorker)
      for (auto device : *devices)  {
        context->world.send(owningRank,
                            device->contextRank(),
                            getFor(device)->tileDescs,
                            tilesOnGPU[device->contextRank()],
                            send_requests[device->contextRank()]);
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
  }
#endif
}

