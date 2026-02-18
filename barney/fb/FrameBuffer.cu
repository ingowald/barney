// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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

  FrameBuffer::FrameBuffer(Context *context,
                           const DevGroup::SP &devices,
                           const bool isOwner)
    : barney_api::FrameBuffer(context),
      isOwner(isOwner),
      devices(devices)
  {
    if (FromEnv::get()->explicitlyDisabled("denoising")) {
      if (context->myRank() == 0)
        std::cout << "#bn: denoising explicitly disabled in env-config." << std::endl;
      enableDenoising = false;
    }
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      getPLD(device)->tiledFB
        = TiledFB::create(device,context->deviceWeNeedToCopyToForFBMap,this);
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
    if (linearNormalChannel) {
      device->rtc->freeMem(linearNormalChannel);
      linearNormalChannel = 0;
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
    out[idx] = ::BARNEY_NS::make_rgba(v);
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
    bool needNormalChannel = (channels & BN_FB_NORMAL) && linearNormalChannel;
    void *colorCopyTarget
      = doDenoising
      ? denoiser->in_rgba
      : linearColorChannel;
    vec3f *normalCopyTarget = nullptr;
    if (doDenoising)
      normalCopyTarget = denoiser->in_normal;
    else if (needNormalChannel)
      normalCopyTarget = (vec3f*)linearNormalChannel;
    BNDataType gatherType
      = doDenoising
      ? BN_FLOAT4
      : colorChannelFormat;

    // this is virtual, and will incur either device copies or mpi
    // pack-gather-unpack
    gatherColorChannel(colorCopyTarget,gatherType,normalCopyTarget);

    // if denoising AND normal channel requested, copy normals from
    // denoiser input to our linear buffer
    if (doDenoising && needNormalChannel) {
      device->rtc->copy(linearNormalChannel, denoiser->in_normal,
                        numPixels.x*numPixels.y*sizeof(vec3f));
    }

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
      // iw - denoiser (currently) doesn't have eaccess to device
      // stream, so runs in default stream --> have to make sure that
      // device stream is synced before we run it.
      device->rtc->sync();
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
    if (!appMemory) return;

    if (!isOwner) {
      // iw 'in theory' apps shoudln't even call map on any rank other
      // thank rank 0, but if they do, let's report them valid black,
      // zero-opacity, depth-infinity frame in case the app wants to
      // do compositing with that.
      int numPixels = this->numPixels.x * this->numPixels.y;
      if (channel == BN_FB_DEPTH) {
        for (int i=0;i<numPixels;i++)
          ((float*)appMemory)[i] = BARNEY_INF;
        return;
      }
      if (channel == BN_FB_DEPTH ||
          channel == BN_FB_PRIMID ||
          channel == BN_FB_INSTID ||
          channel == BN_FB_OBJID) {
        for (int i=0;i<numPixels;i++)
          ((uint32_t*)appMemory)[i] = 0;
        return;
      }
      if (channel == BN_FB_COLOR && requestedFormat == BN_FLOAT4) {
        for (int i=0;i<numPixels;i++)
          ((vec4f*)appMemory)[i] = vec4f(0.f,0.f,0.f,0.f);
        return;
      }
      if (channel == BN_FB_COLOR && requestedFormat == BN_UFIXED8_RGBA) {
        for (int i=0;i<numPixels;i++)
          ((uint32_t*)appMemory)[i] = 0x00000000;
        return;
      }
      return;
    }
    
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

    if (channel == BN_FB_NORMAL && linearNormalChannel) {
      // normals were already linearized during finalizeFrame()
      device->rtc->copy(appMemory,linearNormalChannel,
                        numPixels.x*numPixels.y*sizeof(vec3f));
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
      if (channels & BN_FB_NORMAL)
        linearNormalChannel = rtc->allocMem(np*sizeof(vec3f));

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

