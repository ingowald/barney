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

  RTC_IMPORT_COMPUTE1D(setTileCoords);
  RTC_IMPORT_COMPUTE1D(linearizeColorAndNormal);
  RTC_IMPORT_COMPUTE1D(linearizeAuxChannel);

  struct LinearizeColorAndNormal {
    /* ARGS */
    void      *out_rgba;
    BNDataType colorFormat;
    vec3f     *out_normal;
    float      accumScale;
    TileDesc  *descs;
    AccumTile *tiles;
    vec2i      numPixels;

#if RTC_DEVICE_CODE
    /* CODE */
    inline __rtc_device
    void run(const rtc::ComputeInterface &ci)
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
#endif
  };


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
    SetActiveGPU forDuration(device);
    LinearizeColorAndNormal args={linearColor,colorFormat,
                                  linearNormal,accumScale,
                                  tileDescs,accumTiles,
                                  numPixels};
    linearizeColorAndNormalKernel
      ->launch(numActiveTiles,pixelsPerTile,&args);
  }




  struct LinearizeAuxChannel {
    /* ARGS */
    void           *linearOut;
    vec2i           numPixels;
    AuxChannelTile *aux;
    TileDesc       *descs;

#if RTC_DEVICE_CODE
    /* CODE */
    inline __rtc_device
    void run(const rtc::ComputeInterface &ci)
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
      
      // switch (whichChannel) {
      // case BN_FB_PRIMID: 
      //   ((uint32_t*)linearOut)[idx] = aux.primID[tileIdx].ui[subIdx];
      //   break;
      // case BN_FB_OBJID: 
      //   ((uint32_t*)linearOut)[idx] = aux.objID[tileIdx].ui[subIdx];
      //   break;
      // case BN_FB_INSTID: 
      //   ((uint32_t*)linearOut)[idx] = aux.instID[tileIdx].ui[subIdx];
      //   break;
      // case BN_FB_DEPTH: 
      //   ((float*)linearOut)[idx] = aux.depth[tileIdx].f[subIdx];
      //   break;
      // default:
      //   printf("LinearizeAuxChannel not implemented for channel #%i\n",
      //          whichChannel);
      // }
    }
#endif
  };

  /*! linearize given array's aux tiles, on given device. this can be
      used either for local GPUs on a single node, or on the owner
      after it reveived all worker tiles */
  void TiledFB::linearizeAuxTiles(Device *device,
                                  rtc::ComputeKernel1D *linearizeAuxChannelKernel,
                                  void *linearOut,
                                  vec2i numPixels,
                                  AuxChannelTile *tilesIn,
                                  TileDesc       *descsIn,
                                  int numTiles)
  {
    SetActiveGPU forDuration(device);
    LinearizeAuxChannel args={linearOut,numPixels,
                              tilesIn,descsIn};
    linearizeAuxChannelKernel
      ->launch(numTiles,pixelsPerTile,&args);       
  }
  

    /*! linearize _this gpu's_ channels */
  void TiledFB::linearizeAuxChannel(void *linearChannel,
                                    BNFrameBufferChannel channel)
  {
    AuxChannelTile *aux = 0;
    switch(channel) {
    case BN_FB_DEPTH:
      aux = auxTiles.depth;
      break;
    case BN_FB_PRIMID:
      aux = auxTiles.primID;
      break;
    case BN_FB_INSTID:
      aux = auxTiles.instID;
      break;
    case BN_FB_OBJID:
      aux = auxTiles.objID;
      break;
    default:
      throw std::runtime_error("unsupported aux channel in sending aux!?");
    };
    linearizeAuxTiles(device,linearizeAuxChannelKernel,
                        linearChannel,numPixels,
                        aux,tileDescs,numActiveTiles);
  }




  
  TiledFB::SP TiledFB::create(Device *device, FrameBuffer *owner)
  {
    return std::make_shared<TiledFB>(device, owner);
  }

  TiledFB::TiledFB(Device *device, FrameBuffer *owner)
    : device(device),
      owner(owner)
  {
    setTileCoords
      = createCompute_setTileCoords(device->rtc);
    linearizeColorAndNormalKernel
      = createCompute_linearizeColorAndNormal(device->rtc);
    linearizeAuxChannelKernel
      = createCompute_linearizeAuxChannel(device->rtc);
  }

  TiledFB::~TiledFB()
  {
    free();
    delete setTileCoords;
    delete linearizeColorAndNormalKernel;
  }

  void TiledFB::free()
  {
    SetActiveGPU forDuration(device);
    if (tileDescs) {
      device->rtc->freeMem(tileDescs);
      tileDescs = nullptr;
    }
    if (accumTiles)  {
      device->rtc->freeMem(accumTiles);
      accumTiles = nullptr;
    }
    auto freeTiles = [&](AuxChannelTile *&tiles) {
      if (tiles) {
        device->rtc->freeMem(tiles);
        tiles = 0;
      }
    };
    freeTiles(auxTiles.primID);
    freeTiles(auxTiles.instID);
    freeTiles(auxTiles.objID);
    freeTiles(auxTiles.depth);
  }

  struct SetTileCoords {
    /* kernel ARGS */
    TileDesc *tileDescs;
    int numActiveTiles;
    vec2i numTiles;
    int globalIndex;
    int globalIndexStep;

    /* kernel CODE */
#if RTC_DEVICE_CODE
    inline __rtc_device
    void run(const rtc::ComputeInterface &rtCore);
#endif
  };

#if RTC_DEVICE_CODE
  /* kernel CODE */
  inline __rtc_device
  void SetTileCoords::run(const rtc::ComputeInterface &rtCore)
  {
    int tid
      = rtCore.getThreadIdx().x
      + rtCore.getBlockIdx().x*rtCore.getBlockDim().x;
    if (tid >= numActiveTiles)
      return;
        
    int tileID = tid * globalIndexStep + globalIndex;
        
    int tile_x = tileID % numTiles.x;
    int tile_y = tileID / numTiles.x;
    tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
  }
#endif
  

  void TiledFB::resize(uint32_t channels,
                       vec2i newSize)
  {
    free();
    SetActiveGPU forDuration(device);

    numPixels = newSize;
    numTiles  = divRoundUp(numPixels,vec2i(tileSize));
    numActiveTiles
      = device
      ? divRoundUp(std::max(0,numTiles.x*numTiles.y - device->globalIndex),
                   device->globalIndexStep)
      : 0;
    auto rtc = device->rtc;
    accumTiles
      = (AccumTile *)rtc->allocMem(numActiveTiles * sizeof(AccumTile));
    auto alloc = [&](AuxChannelTile *&tiles) 
    { tiles = (AuxChannelTile *)rtc->allocMem(numActiveTiles*sizeof(*tiles)); };

    if (channels & BN_FB_PRIMID) alloc(auxTiles.primID);
    if (channels & BN_FB_INSTID) alloc(auxTiles.instID);
    if (channels & BN_FB_OBJID)  alloc(auxTiles.objID);
    if (channels & BN_FB_DEPTH)  alloc(auxTiles.depth);
    
    tileDescs
      = (TileDesc *)rtc->allocMem(numActiveTiles * sizeof(TileDesc));
    SetTileCoords args = {
      tileDescs,
      numActiveTiles,
      numTiles,
      device->globalIndex,
      device->globalIndexStep
    };
    if (numActiveTiles > 0)
      setTileCoords
        ->launch(divRoundUp(numActiveTiles,1024),1024,
                 &args);
  }

  // ==================================================================

  RTC_EXPORT_COMPUTE1D(setTileCoords,SetTileCoords);
  RTC_EXPORT_COMPUTE1D(linearizeColorAndNormal,LinearizeColorAndNormal);
  RTC_EXPORT_COMPUTE1D(linearizeAuxChannel,LinearizeAuxChannel);
}




  
