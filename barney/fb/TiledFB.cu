// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "barney/fb/TiledFB.h"
#include "barney/fb/FrameBuffer.h"
#include "barney/common/math.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  // RTC_IMPORT_COMPUTE1D(setTileCoords);
  // RTC_IMPORT_COMPUTE1D(linearizeColorAndNormal);
  // RTC_IMPORT_COMPUTE1D(linearizeAuxChannel);

//   struct LinearizeColorAndNormal {
//     /* ARGS */
//     void      *out_rgba;
//     BNDataType colorFormat;
//     vec3f     *out_normal;
//     float      accumScale;
//     TileDesc  *descs;
//     AccumTile *tiles;
//     vec2i      numPixels;

// #if RTC_DEVICE_CODE
//     /* CODE */
//     inline __rtc_device
//     void run(const rtc::ComputeInterface &ci)
//     {
//       int tileIdx = ci.getBlockIdx().x;
//       TileDesc desc   = descs[tileIdx];
//       AccumTile *tile = &tiles[tileIdx];
      
//       int subIdx = ci.getThreadIdx().x;
//       int iix = subIdx % tileSize;
//       int iiy = subIdx / tileSize;
//       int ix = desc.lower.x + iix;
//       int iy = desc.lower.y + iiy;
//       if (ix >= numPixels.x) return;
//       if (iy >= numPixels.y) return;
//       int idx = ix + numPixels.x*iy;

//       vec4f color = tile->accum[subIdx] * accumScale;
//       if (colorFormat == BN_FLOAT4) 
//         ((vec4f*)out_rgba)[idx] = color;
//       else if (colorFormat == BN_UFIXED8_RGBA) 
//         ((uint32_t*)out_rgba)[idx] = make_rgba(color);
//       else if (colorFormat == BN_UFIXED8_RGBA_SRGB) 
//         ((uint32_t*)out_rgba)[idx] = make_rgba(linear_to_srgb(color));
//       else
//         // unsupported type!?
//         ;
//       if (out_normal)
//         out_normal[idx] = tile->normal[subIdx];
//     }
// #endif
//   };


  __rtc_global
  void linearizeColorAndNormalKernel(const rtc::ComputeInterface &ci,
                                     void      *out_rgba,
                                     BNDataType colorFormat,
                                     vec3f     *out_normal,
                                     float      accumScale,
                                     TileDesc  *descs,
                                     AccumTile *tiles,
                                     vec2i      numPixels)
  {
    int tileIdx = ci.getBlockIdx().x;
    TileDesc desc   = descs[tileIdx];
    AccumTile *tile = &tiles[tileIdx];
      
    int subIdx = ci.getThreadIdx().x;
    int iix = subIdx % tileSize;
    int iiy = subIdx / tileSize;
    int ix = desc.lower.x + iix;
    int iy = desc.lower.y + iiy;
    if (ix >= numPixels.x) return;
    if (iy >= numPixels.y) return;
    int idx = ix + numPixels.x*iy;

    vec4f color = tile->accum[subIdx] * accumScale;
    if (colorFormat == BN_FLOAT4) 
      ((vec4f*)out_rgba)[idx] = color;
    else if (colorFormat == BN_UFIXED8_RGBA) 
      ((uint32_t*)out_rgba)[idx] = make_rgba(color);
    else if (colorFormat == BN_UFIXED8_RGBA_SRGB) 
      ((uint32_t*)out_rgba)[idx] = make_rgba(linear_to_srgb(color));
    else
      // unsupported type!?
      ;
    if (out_normal)
      out_normal[idx] = tile->normal[subIdx];
  }
  

  /*! take this GPU's tiles, and write those tiles' color (and
    optionally normal) channels into the linear frame buffers
    provided. The linearColor is guaranteed to be non-null, and to
    be numPixels.x*numPixels.y vec4fs; linearNormal may be
    null. Linear buffers may live on another GPU, but are
    guaranteed to be on the same node. */
  void TiledFB::linearizeColorAndNormal(void *linearColor,
                                        BNDataType colorFormat,
                                        vec3f *linearNormal,
                                        float  accumScale)
  {
    SetActiveGPU forDuration(appDevice?appDevice:device);
    if (appDevice) 
      appDevice->rtc->copy(appTileDescs,tileDescs,
                           numActiveTilesThisGPU*sizeof(*appTileDescs));
    __rtc_launch(// device
                 (appDevice?appDevice:device)->rtc,
                 // kernel
                 linearizeColorAndNormalKernel,
                 // launch config
                 numActiveTilesThisGPU,pixelsPerTile,
                 // args
                 linearColor,
                 colorFormat,
                 linearNormal,
                 accumScale,
                 appDevice?appTileDescs:tileDescs,
                 appDevice?appAccumTiles:accumTiles,
                 numPixels);
  }



  __rtc_global void linearizeAuxTilesKernel(const rtc::ComputeInterface &ci,
                                            void           *linearOut,
                                            vec2i           numPixels,
                                            AuxChannelTile *aux,
                                            TileDesc       *descs)
  {
    int        tileIdx = ci.getBlockIdx().x;
    TileDesc   desc    = descs[tileIdx];
      
    int subIdx = ci.getThreadIdx().x;
    int iix = subIdx % tileSize;
    int iiy = subIdx / tileSize;
    int ix = desc.lower.x + iix;
    int iy = desc.lower.y + iiy;
    if (ix >= numPixels.x) return;
    if (iy >= numPixels.y) return;
    int idx = ix + numPixels.x*iy;

    ((uint32_t*)linearOut)[idx] = aux[tileIdx].ui[subIdx];
  }

  /*! linearize given array's aux tiles, on given device. this can be
    used either for local GPUs on a single node, or on the owner
    after it reveived all worker tiles */
  void TiledFB::linearizeAuxTiles(Device *device,
                                  void *linearOut,
                                  vec2i numPixels,
                                  AuxChannelTile *tilesIn,
                                  TileDesc       *descsIn,
                                  int numTiles)
  {
    SetActiveGPU forDuration(device);
    __rtc_launch(// device
                 device->rtc,
                 // kernel
                 linearizeAuxTilesKernel,
                 // launchDims
                 numTiles,pixelsPerTile,
                 // args
                 linearOut,
                 numPixels,
                 tilesIn,
                 descsIn);
  }
  

    /*! linearize _this gpu's_ channels */
  void TiledFB::linearizeAuxChannel(void *linearChannel,
                                    BNFrameBufferChannel channel)
  {
    AuxChannelTile *aux = 0;
    switch(channel) {
    case BN_FB_DEPTH:
      aux = appDevice?appAuxTiles.depth:auxTiles.depth;
      break;
    case BN_FB_PRIMID:
      aux = appDevice?appAuxTiles.primID:auxTiles.primID;
      break;
    case BN_FB_INSTID:
      aux = appDevice?appAuxTiles.instID:auxTiles.instID;
      break;
    case BN_FB_OBJID:
      aux = appDevice?appAuxTiles.objID:auxTiles.objID;
      break;
    default:
      throw std::runtime_error("unsupported aux channel in sending aux!?");
    };
    linearizeAuxTiles(appDevice?appDevice:device,
                      linearChannel,numPixels,
                      aux,tileDescs,numActiveTilesThisGPU);
  }

  
  TiledFB::SP TiledFB::create(Device *device,
                              /*! device for the gpu that the app
                                  lives on, if different from current
                                  GPU, and if no peer access is
                                  available */
                              Device *appDevice,
                              FrameBuffer *owner)
  {
    return std::make_shared<TiledFB>(device, appDevice, owner);
  }

  TiledFB::TiledFB(Device *device,
                   Device *appDevice,
                   FrameBuffer *owner)
    : device(device),
      appDevice(appDevice),
      owner(owner)
  {}

  TiledFB::~TiledFB()
  {
    free();
  }

  template<typename T>
  void freeAndSetNull(Device *device, T *&pMem)
  {
    if (pMem) {
      device->rtc->freeMem(pMem);
      pMem = nullptr;
    }
  }
  
  void TiledFB::free()
  {
    SetActiveGPU forDuration(device);
    freeAndSetNull(device,tileDescs);
    freeAndSetNull(device,accumTiles);
    freeAndSetNull(device,auxTiles.primID);
    freeAndSetNull(device,auxTiles.instID);
    freeAndSetNull(device,auxTiles.objID);
    freeAndSetNull(device,auxTiles.depth);

    if (appDevice) {
      SetActiveGPU forDuration(device);
      freeAndSetNull(device,appTileDescs);
      freeAndSetNull(device,appAccumTiles);
      freeAndSetNull(device,appAuxTiles.primID);
      freeAndSetNull(device,appAuxTiles.instID);
      freeAndSetNull(device,appAuxTiles.objID);
      freeAndSetNull(device,appAuxTiles.depth);
    }
  }

  __rtc_global void setTileCoordsKernel(const rtc::ComputeInterface &ci,
                                        TileDesc *tileDescs,
                                        int numActiveTiles,
                                        vec2i numTiles,
                                        int globalIndex,
                                        int globalIndexStep)
  {
    int tid = ci.launchIndex().x;
    if (tid >= numActiveTiles) return;
    
    int tileID = tid * globalIndexStep + globalIndex;
    
    int tile_x = tileID % numTiles.x;
    int tile_y = tileID / numTiles.x;
    tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
  }
  

  void TiledFB::resize(uint32_t channels,
                       vec2i newSize)
  {
    free();
    SetActiveGPU forDuration(device);

    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTilesThisGPU
      = device
      ? divRoundUp(std::max(0,numTiles.x*numTiles.y - device->globalRank()),
                   device->globalSize())
      : 0;
    auto rtc = device->rtc;
    accumTiles
      = (AccumTile *)rtc->allocMem(numActiveTilesThisGPU * sizeof(AccumTile));
    if (appDevice) {
    }
    auto alloc = [&](AuxChannelTile *&tiles) 
    { tiles = (AuxChannelTile *)rtc->allocMem(numActiveTilesThisGPU*sizeof(*tiles)); };

    if (channels & BN_FB_PRIMID) alloc(auxTiles.primID);
    if (channels & BN_FB_INSTID) alloc(auxTiles.instID);
    if (channels & BN_FB_OBJID)  alloc(auxTiles.objID);
    if (channels & BN_FB_DEPTH)  alloc(auxTiles.depth);
    if (appDevice) {
      SetActiveGPU forDuration(appDevice);
      if (channels & BN_FB_PRIMID) alloc(appAuxTiles.primID);
      if (channels & BN_FB_INSTID) alloc(appAuxTiles.instID);
      if (channels & BN_FB_OBJID)  alloc(appAuxTiles.objID);
      if (channels & BN_FB_DEPTH)  alloc(appAuxTiles.depth);
    }
    
    tileDescs
      = (TileDesc *)rtc->allocMem(numActiveTilesThisGPU * sizeof(TileDesc));
    if (appDevice) {
      SetActiveGPU forDuration(appDevice);
      appTileDescs
        = (TileDesc *)rtc->allocMem(numActiveTilesThisGPU * sizeof(TileDesc));
    }
    __rtc_launch(//device
                 device->rtc,
                 // kernel
                 setTileCoordsKernel,
                 // launch config,
                 divRoundUp(numActiveTilesThisGPU,1024),1024,
                 // args
                 tileDescs,
                 numActiveTilesThisGPU,
                 numTiles,
                 device->globalRank(),
                 device->globalSize());
    if (appTileDescs)
      rtc->copy(appTileDescs,tileDescs,numActiveTilesThisGPU * sizeof(TileDesc));
  }

}




  
