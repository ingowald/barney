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
    void                *linearOut;
    BNFrameBufferChannel whichChannel;
    TileDesc  *descs;
    AccumTile *tiles;
    vec2i      numPixels;
      
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
      
      switch (whichChannel) {
      case BN_FB_PRIMID: 
        ((uint32_t*)linearOut)[idx] = tile->primID[subIdx];
        break;
      case BN_FB_OBJID: 
        ((uint32_t*)linearOut)[idx] = tile->objID[subIdx];
        break;
      case BN_FB_INSTID: 
        ((uint32_t*)linearOut)[idx] = tile->instID[subIdx];
        break;
      default:
        printf("LinearizeAuxChannel not implemented for channel #%i\n",
               whichChannel);
      }
    }
  };

  void TiledFB::linearizeAuxChannel(void *linearChannel,
                                    BNFrameBufferChannel whichChannel)
  {
    SetActiveGPU forDuration(device);
    LinearizeAuxChannel args={linearChannel,whichChannel,
                              tileDescs,accumTiles,
                              numPixels};
    linearizeAuxChannelKernel
      ->launch(numActiveTiles,pixelsPerTile,&args);       
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
    if (accumTiles)  {
      device->rtc->freeMem(accumTiles);
      accumTiles = nullptr;
    }
    // if (compressedTiles) {
    //   device->rtc->freeMem(compressedTiles);
    //   compressedTiles = nullptr;
    // }
    if (tileDescs) {
      device->rtc->freeMem(tileDescs);
      tileDescs = nullptr;
    }
  }

  struct SetTileCoords {
    /* kernel ARGS */
    TileDesc *tileDescs;
    int numActiveTiles;
    vec2i numTiles;
    int globalIndex;
    int globalIndexStep;

    /* kernel CODE */
    inline __rtc_device
    void run(const rtc::ComputeInterface &rtCore);
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
  

  void TiledFB::resize(vec2i newSize)
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
    // compressedTiles
    //   = (CompressedTile *)rtc->allocMem(numActiveTiles * sizeof(CompressedTile));
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

#if 0
  struct CompressTiles {
    CompressedTile *compressedTiles;
    AccumTile      *accumTiles;
    float           accumScale;
    int             globalIdx;
    int             globalIdxStep;

    inline __rtc_device
    void run(const rtc::ComputeInterface &rtCore);
  };

#if RTC_DEVICE_CODE
  inline __rtc_device
  void CompressTiles::run(const rtc::ComputeInterface &rtCore)
  { 
    int pixelID = rtCore.getThreadIdx().x;
    int tileID  = rtCore.getBlockIdx().x;

    vec4f color = vec4f(accumTiles[tileID].accum[pixelID])*accumScale;
    vec4f org = color;
    float scale = reduce_max((const vec3f&)color);
    (vec3f&)color *= 1.f/scale;
    compressedTiles[tileID].scale[pixelID] = scale;
    compressedTiles[tileID].normal[pixelID]
      .set(accumTiles[tileID].normal[pixelID]);

      
    uint32_t rgba32
      = make_rgba(color);

    compressedTiles[tileID].rgba[pixelID]
      = rgba32;
  }
#endif
  
  /*! write this tiledFB's tiles into given "compressed" frame buffer
    (i.e., a plain 2D array of numPixels.x*numPixels.y RGBA8
    pixels) */
  void TiledFB::finalizeTiles_launch()
  {
    if (numActiveTiles > 0) {
      SetActiveGPU forDuration(device);
      CompressTiles args = {
        compressedTiles,
        accumTiles,
        1.f/(owner->accumID),
        device->globalIndex,
        device->globalIndexStep,
      };
      device->compressTiles
        ->launch(numActiveTiles,pixelsPerTile,
                 &args);       
    }
  }
  
  RTC_EXPORT_COMPUTE1D(compressTiles,CompressTiles);
#endif
  RTC_EXPORT_COMPUTE1D(setTileCoords,SetTileCoords);
  RTC_EXPORT_COMPUTE1D(linearizeColorAndNormal,LinearizeColorAndNormal);
  RTC_EXPORT_COMPUTE1D(linearizeAuxChannel,LinearizeAuxChannel);
}




  
