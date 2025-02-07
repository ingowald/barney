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

#include "barney/common/barney-common.h"
#include "barney/fb/FrameBuffer.h"
// #include <cuda_runtime.h>
#if BARNEY_HAVE_OIDN
# include <OpenImageDenoise/oidn.h>
#endif

namespace barney {

  // inline __device__ float saturate(float f, float lo=0.f, float hi=1.f)
  // { return max(lo,min(f,hi)); }
  
  inline __both__ float from_8bit(uint8_t v) {
    return float(v) * (1.f/255.f);
  }
  
  inline __both__ vec4f from_8bit(uint32_t v) {
    return vec4f(from_8bit(uint8_t((v >> 0)&0xff)),
                 from_8bit(uint8_t((v >> 8)&0xff)),
                 from_8bit(uint8_t((v >> 16)&0xff)),
                 from_8bit(uint8_t((v >> 24)&0xff)));
  }
  
  inline __both__ uint32_t _make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __both__ uint32_t make_rgba8(const vec4f color)
  {
    uint32_t r = _make_8bit(color.x);
    uint32_t g = _make_8bit(color.y);
    uint32_t b = _make_8bit(color.z);
    uint32_t a = 0xff; //make_8bit(color.w);
    uint32_t ret =
      (r <<  0) |
      (g <<  8) |
      (b << 16) |
      (a << 24);
    return ret;
  }
  
  FrameBuffer::FrameBuffer(Context *context,
                           const DevGroup::SP &devices,
                           const bool isOwner)
    : SlottedObject(context,devices),
      isOwner(isOwner)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      getPLD(device)->tiledFB
        = TiledFB::create(device,this);
    }

    Device *device = getDenoiserDevice();
    assert(device);
    denoiser = device->rtc->createDenoiser();
  }

  FrameBuffer::~FrameBuffer()
  {
    freeResources();
    denoiser = 0;
  }

  bool FrameBuffer::set1i(const std::string &member, const int &value)
  {
    if (member == "showCrosshairs") {
      showCrosshairs = value;
      return true;
    }
    return false;
  }

  void FrameBuffer::freeResources()
  {
    Device *device = getDenoiserDevice();
    if (denoisedColor) {
      // BARNEY_CUDA_CALL(Free(denoisedColor));
      device->rtc->freeMem(denoisedColor);
      denoisedColor = 0;
    }
    if (linearColor) {
      // BARNEY_CUDA_CALL(Free(linearColor));
      device->rtc->freeMem(linearColor);
      linearColor = 0;
    }
    // if (linearAlpha) {
    //   device->rtc->freeMem(linearAlpha);
    //   // BARNEY_CUDA_CALL(Free(linearAlpha));
    //   linearAlpha = 0;
    // }
    if (linearDepth) {
      device->rtc->freeMem(linearDepth);
      // BARNEY_CUDA_CALL(Free(linearDepth));
      linearDepth = 0;
    }
    if (linearNormal) {
      device->rtc->freeMem(linearNormal);
      // BARNEY_CUDA_CALL(Free(linearNormal));
      linearNormal = 0;
    }
  }

// #if 1
  struct ToFixed8 {
    uint32_t *out;
    vec4f *in;
    vec2i numPixels;
    bool SRGB;
    template<typename CI>
    __both__ void run(const CI &ci)
    {
      int ix = ci.getThreadIdx().x+ci.getBlockIdx().x*ci.getBlockDim().x;
      if (ix >= numPixels.x) return;
      int iy = ci.getThreadIdx().y+ci.getBlockIdx().y*ci.getBlockDim().y;
      if (iy >= numPixels.y) return;
      int idx = ix+numPixels.x*iy;
      vec4f v = in[idx];
      v.x = clamp(v.x);
      v.y = clamp(v.y);
      v.z = clamp(v.z);
      if (SRGB) {
        // this doesn't make sense - the color channel has ALREADY been
        // gamma-corrected in tonemap()!?
        v.x = linear_to_srgb(v.x);
        v.y = linear_to_srgb(v.y);
        v.z = linear_to_srgb(v.z);
      }
      out[idx] = make_rgba(v);
    }
  };
// #else
//   template<bool SRGB>
//   __global__
//   void toFixed8(uint32_t *out,
//                 vec4f *in,
//                 vec2i numPixels)
//   {
//     int ix = threadIdx.x+blockIdx.x*blockDim.x;
//     if (ix >= numPixels.x) return;
//     int iy = threadIdx.y+blockIdx.y*blockDim.y;
//     if (iy >= numPixels.y) return;
//     int idx = ix+numPixels.x*iy;

//     vec4f v = in[idx];
//     v.x = clamp(v.x);
//     v.y = clamp(v.y);
//     v.z = clamp(v.z);
//     if (SRGB) {
//       // this doesn't make sense - the color channel has ALREADY been
//       // gamma-corrected in tonemap()!?
//       v.x = linear_to_srgb(v.x);
//       v.y = linear_to_srgb(v.y);
//       v.z = linear_to_srgb(v.z);
//     }
//     out[idx] = make_rgba(v);
//     // out[idx] = make_rgba8(v);
//   }
// #endif

// #if 1
  struct ToneMap {
    vec4f *color;
    vec2i numPixels;
    
    template<typename CI>
    __both__ void run(const CI &ci)
    {
      int ix = ci.getThreadIdx().x+ci.getBlockIdx().x*ci.getBlockDim().x;
      if (ix >= numPixels.x) return;
      int iy = ci.getThreadIdx().y+ci.getBlockIdx().y*ci.getBlockDim().y;
      if (iy >= numPixels.y) return;
      int idx = ix+numPixels.x*iy;
      
      vec4f v = color[idx];
#if 0
      v.x = linear_to_srgb(v.x);
      v.y = linear_to_srgb(v.y);
      v.z = linear_to_srgb(v.z);
#elif 1
      v.x = sqrtf(v.x);
      v.y = sqrtf(v.y);
      v.z = sqrtf(v.z);
#else
      // v.x = linear_to_srgb(v.x);
      // v.y = linear_to_srgb(v.y);
      // v.z = linear_to_srgb(v.z);
#endif
      color[idx] = v;
    }
  };
// #else
//   __global__ void toneMap(vec4f *color, vec2i numPixels)
//   {
//     int ix = threadIdx.x+blockIdx.x*blockDim.x;
//     if (ix >= numPixels.x) return;
//     int iy = threadIdx.y+blockIdx.y*blockDim.y;
//     if (iy >= numPixels.y) return;
//     int idx = ix+numPixels.x*iy;

//     vec4f v = color[idx];
// #if 0
//     v.x = linear_to_srgb(v.x);
//     v.y = linear_to_srgb(v.y);
//     v.z = linear_to_srgb(v.z);
// #elif 1
//     v.x = sqrtf(v.x);
//     v.y = sqrtf(v.y);
//     v.z = sqrtf(v.z);
// #else
//     // v.x = linear_to_srgb(v.x);
//     // v.y = linear_to_srgb(v.y);
//     // v.z = linear_to_srgb(v.z);
// #endif
//     color[idx] = v;
//   }
// #endif

  void FrameBuffer::finalizeTiles()
  {
    for (auto device : *devices) 
      getFor(device)->finalizeTiles_launch();
    for (auto device : *devices)
      device->sync();
  }
  
  void FrameBuffer::finalizeFrame()
  {
    dirty = true;
    ownerGatherCompressedTiles();
    if (isOwner) {
      unpackTiles();
    }
  }

  // __global__ void g_unpackTiles(vec2i numPixels,
  //                               vec3f *colors,
  //                               float *alphas,
  //                               vec3f *normals,
  //                               float *depths,
  //                               CompressedTile *tiles,
  //                               TileDesc *descs)
  // {
  //   int tileIdx = blockIdx.x;

  //   const CompressedTile &tile = tiles[tileIdx];
  //   const TileDesc        desc = descs[tileIdx];
    
  //   int subIdx = threadIdx.x;
  //   int iix = subIdx % tileSize;
  //   int iiy = subIdx / tileSize;
  //   int ix = desc.lower.x + iix;
  //   int iy = desc.lower.y + iiy;
  //   if (ix >= numPixels.x) return;
  //   if (iy >= numPixels.y) return;
  //   int idx = ix + numPixels.x*iy;

  //   uint32_t rgba8 = tile.rgba[subIdx];
  //   vec4f rgba = from_8bit(rgba8);
  //   float alpha = rgba.w;
  //   float scale = float(tile.scale[subIdx]);
  //   vec3f color = vec3f(rgba.x,rgba.y,rgba.z)*scale;
  //   vec3f normal = tile.normal[subIdx].get3f();
  //   float depth = tile.depth[subIdx];

  //   colors[idx] = color;
  //   alphas[idx] = alpha;
  //   depths[idx] = depth;
  //   normals[idx] = normal;
  // }
  
  
  struct UnpackTiles {
    vec2i numPixels;
    float4 *out_rgba;
    vec3f *normals;
    float *depths;
    CompressedTile *tiles;
    TileDesc *descs;
    
    template<typename CI>
    __both__ void run(const CI &ci)
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
      float depth = tile.depth[subIdx];

      // auto checkFragComp = [](float f) {
      //   if (isnan(f)) printf("NAN fragment!\n");
      //   if (isinf(f)) printf("INF fragment!\n");
      // };
      // checkFragComp(rgba.x);
      // checkFragComp(rgba.y);
      // checkFragComp(rgba.z);
      
      
      out_rgba[idx] = (const float4&)rgba;//color;
      depths[idx] = depth;
      normals[idx] = normal;
    }
  };
  
  void FrameBuffer::unpackTiles()
  {
    // #if 1
    UnpackTiles args = {
      numPixels,
      (float4*)linearColor,
      linearNormal,
      linearDepth,
      gatheredTilesOnOwner.compressedTiles,
      gatheredTilesOnOwner.tileDescs
    };
    auto device = getDenoiserDevice();
    device->unpackTiles->launch(gatheredTilesOnOwner.numActiveTiles,
                                pixelsPerTile,
                                &args);
    device->sync();
    //     CHECK_CUDA_LAUNCH(g_unpackTiles,
    //                       //
    //                       gatheredTilesOnOwner.numActiveTiles,pixelsPerTile,0,0,
    //                       //
    //                       numPixels,
    //                       linearColor,
    //                       linearAlpha,
    //                       linearNormal,
    //                       linearDepth,
    //                       gatheredTilesOnOwner.compressedTiles,
    //                       gatheredTilesOnOwner.tileDescs);
    // #else
    //     g_unpackTiles<<<gatheredTilesOnOwner.numActiveTiles,pixelsPerTile>>>
    //       (numPixels,
    //        linearColor,
    //        linearAlpha,
    //        linearNormal,
    //        linearDepth,
    //        gatheredTilesOnOwner.compressedTiles,
    //        gatheredTilesOnOwner.tileDescs);
    // #endif
  }

  void FrameBuffer::read(BNFrameBufferChannel channel,
                         void *hostPtr,
                         BNDataType requestedFormat)
  {
    if (!isOwner) return;

    Device *device = getDenoiserDevice();

    if (dirty) {
      
      // -----------------------------------------------------------------------------
      // (HDR) denoising
      // -----------------------------------------------------------------------------
      if (!denoiser) {
        device->rtc->copy(this->denoisedColor,this->linearColor,
                          numPixels.x*numPixels.y*sizeof(vec4f));
      } else {
        float blendFactor = accumID / (accumID+200.f); 
        denoiser->run(this->denoisedColor,
                      this->linearColor,
                      this->linearNormal,blendFactor);
      }

      // -----------------------------------------------------------------------------
      // tone map (denoised) frame buffer
      // -----------------------------------------------------------------------------
      {
        vec2i bs(8,8);
        ToneMap args = { denoisedColor,numPixels };
        device->toneMap->launch(divRoundUp(numPixels,bs),bs,
                                &args);
      }
      dirty = false;
    }
    if (channel == BN_FB_DEPTH && hostPtr && linearDepth) {
      if (requestedFormat != BN_FLOAT)
        throw std::runtime_error("can only read depth channel as BN_FLOAT format");
      if (!linearDepth)
        throw std::runtime_error("requesting to read depth channel, but didn't create one");
      device->rtc->copy(hostPtr,linearDepth,
                        numPixels.x*numPixels.y*sizeof(float));
      device->rtc->sync();
      return;
    }

    if (!hostPtr) return;
    
    if (channel != BN_FB_COLOR)
      throw std::runtime_error("trying to read un-known channel!?");

    switch(requestedFormat) {
    case BN_FLOAT4: 
    case BN_FLOAT4_RGBA: {
      device->rtc->copy(hostPtr,denoisedColor,
                        numPixels.x*numPixels.y*sizeof(vec4f));
    } break;
    case BN_UFIXED8_RGBA: {
      uint32_t *asFixed8
        = (uint32_t*)device->rtc->allocMem(numPixels.x*numPixels.y*sizeof(uint32_t));
      vec2i bs(8,8);
      ToFixed8 args = { asFixed8,denoisedColor,numPixels,false };
      device->toFixed8->launch(divRoundUp(numPixels,bs),bs,&args);
      device->rtc->copy(hostPtr,asFixed8,numPixels.x*numPixels.y*sizeof(uint32_t));
      device->rtc->freeMem(asFixed8);
    } break;
    case BN_UFIXED8_RGBA_SRGB: {
      uint32_t *asFixed8
        = (uint32_t*)device->rtc->allocMem(numPixels.x*numPixels.y*sizeof(uint32_t));
      vec2i bs(8,8);
      ToFixed8 args = { asFixed8,denoisedColor,numPixels,true };
      device->toFixed8->launch(divRoundUp(numPixels,bs),bs,&args);
      device->rtc->copy(hostPtr,asFixed8,numPixels.x*numPixels.y*sizeof(uint32_t));
      device->rtc->freeMem(asFixed8);
    } break;
    default:
      throw std::runtime_error("requested to read color channel in un-supported format #"
                               +std::to_string((int)requestedFormat));
    };
    device->rtc->sync();
  }
  
  void FrameBuffer::resize(vec2i size,
                           uint32_t channels)
  {
    for (auto device : *devices)
      getFor(device)->resize(size);
    
    freeResources();
    numPixels = size;

    if (isOwner) {
      auto rtc = getDenoiserDevice()->rtc;
      int np = numPixels.x*numPixels.y;
      denoisedColor = (vec4f*)rtc->allocMem(np*sizeof(*denoisedColor));
      linearDepth   = (float *)rtc->allocMem(np*sizeof(*linearDepth));
      linearColor   = (vec4f *)rtc->allocMem(np*sizeof(*linearColor));
      linearNormal  = (vec3f *)rtc->allocMem(np*sizeof(*linearNormal));
      
      // if (!denoiser) denoiser = Denoiser::create(this);
      // denoiser->resize();
      if (denoiser)
        denoiser->resize(numPixels);
    }
  }
    
  FrameBuffer::PLD *FrameBuffer::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  
  TiledFB *FrameBuffer::getFor(Device *device)
  {
    auto pld = getPLD(device);
    assert(pld);
    return pld->tiledFB.get();
  }

  Device *FrameBuffer::getDenoiserDevice() const
  {
    return (*devices)[0];
  }
  
}
  
// RTC_DECLARE_COMPUTE(copyPixels,barney::CopyPixels);
RTC_DECLARE_COMPUTE(toneMap,barney::ToneMap);
RTC_DECLARE_COMPUTE(toFixed8,barney::ToFixed8);
RTC_DECLARE_COMPUTE(unpackTiles,barney::UnpackTiles);
