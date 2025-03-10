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
#if BARNEY_HAVE_OIDN
# include <OpenImageDenoise/oidn.h>
#endif

namespace BARNEY_NS {

  inline __rtc_device float from_8bit(uint8_t v) {
    return float(v) * (1.f/255.f);
  }
  
  inline __rtc_device vec4f from_8bit(uint32_t v) {
    return vec4f(from_8bit(uint8_t((v >> 0)&0xff)),
                 from_8bit(uint8_t((v >> 8)&0xff)),
                 from_8bit(uint8_t((v >> 16)&0xff)),
                 from_8bit(uint8_t((v >> 24)&0xff)));
  }
  
  inline __rtc_device uint32_t _make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __rtc_device uint32_t make_rgba8(const vec4f color)
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
    : barney_api::FrameBuffer(context),
      //SlottedObject(context,devices),
      isOwner(isOwner),
      devices(devices)
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
      device->rtc->freeMem(denoisedColor);
      denoisedColor = 0;
    }
    if (linearColor) {
      device->rtc->freeMem(linearColor);
      linearColor = 0;
    }
    if (linearDepth) {
      device->rtc->freeMem(linearDepth);
      linearDepth = 0;
    }
    if (linearNormal) {
      device->rtc->freeMem(linearNormal);
      linearNormal = 0;
    }
  }

  struct ToFixed8 {
    uint32_t *out;
    vec4f *in;
    vec2i numPixels;
    bool SRGB;
    __rtc_device void run(const rtc::ComputeInterface &ci);
  };

#if RTC_DEVICE_CODE
  __rtc_device void ToFixed8::run(const rtc::ComputeInterface &ci)
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
#endif

  struct ToneMap {
    vec4f *color;
    vec2i numPixels;
    
#if RTC_DEVICE_CODE
    __rtc_device void run(const rtc::ComputeInterface &ci);
#endif
  };
  
#if RTC_DEVICE_CODE
  __rtc_device void ToneMap::run(const rtc::ComputeInterface &ci)
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
    /* nothing - leave as is */
#endif
    color[idx] = v;
  }
#endif

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

  struct UnpackTiles {
    vec2i numPixels;
    rtc::float4 *out_rgba;
    vec3f *normals;
    float *depths;
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
    float depth = tile.depth[subIdx];

    out_rgba[idx] = (const rtc::float4&)rgba;
    depths[idx] = depth;
    normals[idx] = normal;
  }
#endif
  
  void FrameBuffer::unpackTiles()
  {
    UnpackTiles args = {
      numPixels,
      (rtc::float4*)linearColor,
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
  }

  /*! "finalize" and read the frame buffer. If this function gets
      called with a null hostPtr we will still finalize the frame
      buffer and run the denoiser, just not copy it to host; the
      result can then be read by framebuffergetpointer() */
  void FrameBuffer::read(BNFrameBufferChannel channel,
                         void *hostPtr,
                         BNDataType requestedFormat)
  {
    if (!isOwner) return;

    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);
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
        vec2ui bs(8,8);
        ToneMap args = { denoisedColor,numPixels };
        device->toneMap->launch(divRoundUp(vec2ui(numPixels),bs),bs,
                                &args);
      }
      dirty = false;
    }

    if (!hostPtr) {
      device->rtc->sync();
      return;
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
      vec2ui bs(8,8);
      ToFixed8 args = { asFixed8,denoisedColor,numPixels,false };
      device->toFixed8->launch(divRoundUp(vec2ui(numPixels),bs),bs,&args);
      //BARNEY_CUDA_SYNC_CHECK();
      device->rtc->copy(hostPtr,asFixed8,numPixels.x*numPixels.y*sizeof(uint32_t));
      device->rtc->freeMem(asFixed8);
      //BARNEY_CUDA_SYNC_CHECK();
    } break;
    case BN_UFIXED8_RGBA_SRGB: {
      uint32_t *asFixed8
        = (uint32_t*)device->rtc->allocMem(numPixels.x*numPixels.y*sizeof(uint32_t));
      vec2ui bs(8,8);
      ToFixed8 args = { asFixed8,denoisedColor,numPixels,true };
      device->toFixed8->launch(divRoundUp(vec2ui(numPixels),bs),bs,&args);
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
      denoisedColor = (vec4f *)rtc->allocMem(np*sizeof(*denoisedColor));
      linearDepth   = (float *)rtc->allocMem(np*sizeof(*linearDepth));
      linearColor   = (vec4f *)rtc->allocMem(np*sizeof(*linearColor));
      linearNormal  = (vec3f *)rtc->allocMem(np*sizeof(*linearNormal));
      
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

  void *FrameBuffer::getPointer(BNFrameBufferChannel channel)
  {
    switch(channel) {
    case BN_FB_COLOR:
      return denoisedColor;
    case BN_FB_DEPTH:
      return linearDepth;
    default:
      BARNEY_NYI();
    };
  }
  
  Device *FrameBuffer::getDenoiserDevice() const
  {
    return (*devices)[0];
  }
  
  RTC_EXPORT_COMPUTE2D(toneMap,ToneMap);
  RTC_EXPORT_COMPUTE2D(toFixed8,ToFixed8);
  RTC_EXPORT_COMPUTE1D(unpackTiles,UnpackTiles);
}
  
