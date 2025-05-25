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
#include "barney/common/math.h"
#include "barney/common/Data.h"
#include "barney/fb/FrameBuffer.h"
#if BARNEY_HAVE_OIDN
# include <OpenImageDenoise/oidn.h>
#endif
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  RTC_IMPORT_COMPUTE2D(linearToFixed8);

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
    linear_toFixed8 = createCompute_linearToFixed8(device->rtc);
  }

  FrameBuffer::~FrameBuffer()
  {
    freeResources();
    denoiser = 0;
  }

  bool FrameBuffer::needHitIDs() const
  {
    return channels & (BN_FB_PRIMID|BN_FB_INSTID|BN_FB_OBJID);
  }

  bool FrameBuffer::set1i(const std::string &member, const int &value)
  {
    if (member == "showCrosshairs") {
      showCrosshairs = value;
      return true;
    }
    if (member == "enableDenoising") {
      enableDenoising = value;
      return true;
    }
    return false;
  }

  void FrameBuffer::freeResources()
  {
    Device *device = getDenoiserDevice();
    if (linearColorChannel) {
      device->rtc->freeMem(linearColorChannel);
      linearColorChannel = 0;
    }
    if (linearAuxChannel) {
      device->rtc->freeMem(linearAuxChannel);
      linearAuxChannel = 0;
    }
  }

  struct LinearToFixed8 {
    uint32_t *out;
    vec4f    *in;
    vec2i numPixels;
    bool SRGB;
#if RTC_DEVICE_CODE
    __rtc_device void run(const rtc::ComputeInterface &ci);
#endif
  };

#if RTC_DEVICE_CODE
  __rtc_device void LinearToFixed8::run(const rtc::ComputeInterface &ci)
  {
    int ix = ci.getThreadIdx().x+ci.getBlockIdx().x*ci.getBlockDim().x;
    if (ix >= numPixels.x) return;
    int iy = ci.getThreadIdx().y+ci.getBlockIdx().y*ci.getBlockDim().y;
    if (iy >= numPixels.y) return;
    int idx = ix+numPixels.x*iy;
    vec4f v = in[idx];
    v = saturate(v);
    if (SRGB)
      (vec3f&)v = linear_to_srgb((vec3f&)v);
    out[idx] = make_rgba(v);
  }
#endif

  void FrameBuffer::finalizeTiles()
  {}

  void FrameBuffer::finalizeFrame()
  {
    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);

    /* first, figure out whether we do denoising, and where to write
       color and normals to. fi we do denoisign this will (ahve to) go
       into the respective inputs of the denoiser; otherwise, we write
       color directly into our linearbuffer, and won't write normal at
       all */
    bool doDenoising = (denoiser != 0) && enableDenoising;
    void *colorCopyTarget
      = doDenoising
      ? denoiser->in_rgba
      : linearColorChannel;
    vec3f *normalCopyTarget
      = doDenoising
      ? denoiser->in_normal
      : nullptr;
    BNDataType gatherType
      = doDenoising
      ? BN_FLOAT4
      : colorChannelFormat;
    // this is virtual, and will incur either device copies or mpi
    // pack-gather-unpack
    gatherColorChannel(colorCopyTarget,gatherType,normalCopyTarget);
    if (channels & BN_FB_DEPTH)
      gatherAuxChannel(BN_FB_DEPTH);
    if (channels & BN_FB_PRIMID)
      gatherAuxChannel(BN_FB_PRIMID);
    if (channels & BN_FB_OBJID)
      gatherAuxChannel(BN_FB_OBJID);
    if (channels & BN_FB_INSTID)
      gatherAuxChannel(BN_FB_INSTID);
  }

  /*! gather color (and normal, if required for denoising),
    (re-)format into a linear buffer, perform denoising (if
    required), convert to requested format, and copy to the
    application pointed provided */
  void FrameBuffer::readColorChannel(void *appMemory,
                                     BNDataType requestedFormat)
  {
    assert(requestedFormat == colorChannelFormat);
    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);

    /* first, figure out whether we do denoising, and where to write
       color and normals to. fi we do denoisign this will (ahve to) go
       into the respective inputs of the denoiser; otherwise, we write
       color directly into our linearbuffer, and won't write normal at
       all */
    bool doDenoising = denoiser != 0 && enableDenoising;


    if (doDenoising) {
      /* run denoiser - this will write pixels in float4 format to
         denoiser->out_rgba */
      float blendFactor = (accumID-1) / (accumID+100.f);
      denoiser->run(blendFactor);

      switch(requestedFormat) {
      case BN_FLOAT4: {
        device->rtc->copy(appMemory,denoiser->out_rgba,
                          numPixels.x*numPixels.y*sizeof(vec4f));
      } break;
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB: {
        bool srgb = (requestedFormat == BN_UFIXED8_RGBA_SRGB);
        vec2ui bs(8,8);
        LinearToFixed8 args = {
          (uint32_t*)linearColorChannel,denoiser->out_rgba,numPixels,srgb
        };
        linear_toFixed8->launch(divRoundUp(vec2ui(numPixels),bs),bs,&args);
        device->rtc->copy(appMemory,linearColorChannel,
                          numPixels.x*numPixels.y*sizeof(uint32_t));
      } break;
      default:
        throw std::runtime_error
          ("requested to read color channel in un-supported format #"
           +std::to_string((int)requestedFormat));
      };
    } else {
      size_t sizeOfPixel
        = (requestedFormat == BN_FLOAT4)
        ? sizeof(vec4f)
        : sizeof(uint32_t);
      device->rtc->copy(appMemory,linearColorChannel,
                        numPixels.x*numPixels.y*sizeOfPixel);
    }
    device->rtc->sync();
  }

  /*! "finalize" and read the frame buffer. If this function gets
      called with a null hostPtr we will still finalize the frame
      buffer and run the denoiser, just not copy it to host; the
      result can then be read by framebuffergetpointer() */
  void FrameBuffer::read(BNFrameBufferChannel channel,
                         void *appMemory,
                         BNDataType requestedFormat)
  {
    if (!isOwner) return;
    if (!appMemory) return;

    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);

    if (channel == BN_FB_COLOR) {
      readColorChannel(appMemory,requestedFormat);
      return;
    }

    if (channel == BN_FB_DEPTH ||
        channel == BN_FB_PRIMID ||
        channel == BN_FB_INSTID ||
        channel == BN_FB_OBJID) {
      writeAuxChannel(linearAuxChannel,channel);
      // NOTE: depth + id buffers happen to be the same bytes-per-pixel
      device->rtc->copy(appMemory,linearAuxChannel,
                        numPixels.x*numPixels.y*sizeof(uint32_t));
      return;
    }

    throw std::runtime_error("un-handled frame buffer channel/format combination "
                             +to_string(channel)
                             +" "+to_string(requestedFormat));
  }

  void FrameBuffer::resize(BNDataType colorFormat,
                           vec2i size,
                           uint32_t channels)
  {
    this->channels = channels;
    this->colorChannelFormat = colorFormat;

    for (auto device : *devices)
      getFor(device)->resize(channels,size);

    freeResources();
    numPixels = size;

    size_t sizeOfPixel
      = (colorFormat == BN_FLOAT4)
      ? sizeof(vec4f)
      : sizeof(uint32_t);

    if (isOwner) {
      auto rtc = getDenoiserDevice()->rtc;
      int np = numPixels.x*numPixels.y;
      linearColorChannel = rtc->allocMem(np*sizeOfPixel);
      linearAuxChannel   = rtc->allocMem(np*sizeof(uint32_t));

      if (denoiser)
        denoiser->resize(numPixels);
    }
  }

  FrameBuffer::PLD *FrameBuffer::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
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

  RTC_EXPORT_COMPUTE2D(linearToFixed8,LinearToFixed8);
}

