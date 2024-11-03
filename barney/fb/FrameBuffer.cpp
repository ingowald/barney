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

#include "barney/fb/FrameBuffer.h"
#if BARNEY_HAVE_OIDN
# include <OpenImageDenoise/oidn.h>
#endif

namespace barney {

  struct DenoiserNone : public Denoiser {
    DenoiserNone(FrameBuffer *fb);
    virtual ~DenoiserNone();
    void resize() override;
    void run() override;
  };

#if !BARNEY_DISABLE_DENOISING
  struct DenoiserOIDN : public Denoiser {
    DenoiserOIDN(FrameBuffer *fb)
      : Denoiser(fb)
    {
      device = oidnNewDevice(OIDN_DEVICE_TYPE_CUDA);
      oidnCommitDevice(device);
      filter = oidnNewFilter(device,"RT");
    }
    virtual ~DenoiserOIDN()
    {
      if (colorBuf)  oidnReleaseBuffer(colorBuf);
      if (normalBuf) oidnReleaseBuffer(normalBuf);
      if (outputBuf) oidnReleaseBuffer(outputBuf);
      if (denoiserInput)
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserInput));
      if (denoiserAlpha)
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserAlpha));
      if (denoiserOutput)
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserOutput));
      if (denoiserNormal)
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserNormal));
    }      
    void resize() override
    {
      Denoiser::resize();
      vec2i numPixels = fb->numPixels;
      if (denoiserInput)
        BARNEY_CUDA_CALL(Free(denoiserInput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserInput,
                              numPixels.x*numPixels.y*sizeof(*denoiserInput)));
    
      if (denoiserAlpha)
        BARNEY_CUDA_CALL(Free(denoiserAlpha));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserAlpha,
                              numPixels.x*numPixels.y*sizeof(*denoiserAlpha)));
      if (denoiserOutput)
        BARNEY_CUDA_CALL(Free(denoiserOutput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(*denoiserOutput)));
      
      if (denoiserNormal)
        BARNEY_CUDA_CALL(Free(denoiserNormal));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserNormal,
                              numPixels.x*numPixels.y*sizeof(*denoiserNormal)));
      if (colorBuf)  oidnReleaseBuffer(colorBuf);
      if (normalBuf) oidnReleaseBuffer(normalBuf);
      if (outputBuf) oidnReleaseBuffer(outputBuf);
      colorBuf
        = oidnNewSharedBuffer(device, denoiserInput,
                              numPixels.x*numPixels.y*sizeof(*denoiserInput));
      normalBuf
        = oidnNewSharedBuffer(device, denoiserNormal,
                              numPixels.x*numPixels.y*sizeof(*denoiserNormal));
      outputBuf
        = oidnNewSharedBuffer(device, denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(*denoiserOutput));
      oidnSetFilterImage(filter,"color",colorBuf,
                         OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      oidnSetFilterImage(filter,"normal",normalBuf,
                         OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      oidnSetFilterImage(filter,"output",outputBuf,
                         OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      oidnSetFilterBool(filter,"hdr",true);
      oidnCommitFilter(filter);
    }
    void run() override
    {
      oidnExecuteFilter(filter);
      const char *error;
      oidnGetDeviceError(device,&error);
      if (error)
        PRINT(error);
    }
    
    float3              *denoiserInput   = 0;
    float               *denoiserAlpha   = 0;
    float3              *denoiserOutput  = 0;
    float3              *denoiserNormal  = 0;
    
    OIDNBuffer outputBuf = 0; 
    OIDNBuffer normalBuf = 0; 
    OIDNBuffer colorBuf = 0; 
    OIDNDevice device = 0;
    OIDNFilter filter = 0;
  };

#if OPTIX_VERSION > 80000  
  struct DenoiserOptix : public Denoiser {
    DenoiserOptix(FrameBuffer *fb)
      : Denoiser(fb)
    {
      denoiserOptions.guideAlbedo = 0;
      denoiserOptions.guideNormal = 1;
      denoiserOptions.denoiseAlpha
        = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
        
      auto device = fb->context->getDevice(0);

      OptixDeviceContext optixContext
        = owlContextGetOptixContext(/*device->owl*/
                                    context->globalContextAcrossAllGPUs,0);
      optixDenoiserCreate(optixContext,
                          OPTIX_DENOISER_MODEL_KIND_HDR,
                          &denoiserOptions,
                          &denoiser);
    }      
    virtual ~DenoiserOptix();
    void resize() override
    {
      Denoiser::resize();
      vec2i numPixels = fb->numPixels;
      if (denoiserInput)
        BARNEY_CUDA_CALL(Free(denoiserInput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserInput,
                              numPixels.x*numPixels.y*sizeof(*denoiserInput)));
    
      if (denoiserOutput)
        BARNEY_CUDA_CALL(Free(denoiserOutput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(*denoiserOutput)));
      
      if (denoiserNormal)
        BARNEY_CUDA_CALL(Free(denoiserNormal));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserNormal,
                              numPixels.x*numPixels.y*sizeof(*denoiserNormal)));
      denoiserSizes.overlapWindowSizeInPixels = 0;
      // PING; BARNEY_CUDA_SYNC_CHECK();
      optixDenoiserComputeMemoryResources(/*const OptixDenoiser */
                                          denoiser,
                                          // unsigned int        outputWidth,
                                          numPixels.x,
                                          // unsigned int        outputHeight,
                                          numPixels.y,
                                          // OptixDenoiserSizes* returnSizes
                                          &denoiserSizes
                                          );
      if (denoiserScratch) {
        BARNEY_CUDA_CALL(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch,
                              denoiserSizes.withoutOverlapScratchSizeInBytes));
      if (denoiserState) {
        BARNEY_CUDA_CALL(Free(denoiserState));
        denoiserState = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserState,
                              denoiserSizes.stateSizeInBytes));
      optixDenoiserSetup(// OptixDenoiser denoiser,
                         denoiser,
                         // CUstream      stream,
                         0,//device->launchStream,
                         // unsigned int  inputWidth,
                         numPixels.x,
                         // unsigned int  inputHeight,
                         numPixels.y,
                         // CUdeviceptr   denoiserState,
                         (CUdeviceptr)denoiserState,
                         // size_t        denoiserStateSizeInBytes,
                         denoiserSizes.stateSizeInBytes,
                         // CUdeviceptr   scratch,
                         (CUdeviceptr)denoiserScratch,
                         //size_t        scratchSizeInBytes
                         denoiserSizes.withoutOverlapScratchSizeInBytes
                         );
    }
    void run() override
    {
      OptixDenoiserGuideLayer guideLayer = {};
      OptixDenoiserLayer layer = {};
      layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
      layer.input.rowStrideInBytes = numPixels.x*sizeof(float4);
      layer.input.pixelStrideInBytes = sizeof(float4);
      layer.input.width = numPixels.x;
      layer.input.height = numPixels.y;
      layer.input.data = (CUdeviceptr)denoiserInput;

      guideLayer.normal = layer.input;
      guideLayer.normal.data = (CUdeviceptr)denoiserNormal;
      layer.output = layer.input;
      layer.output.data = (CUdeviceptr)denoiserOutput;

      OptixDenoiserParams denoiserParams = {};

      optixDenoiserInvoke
        (
         denoiser,
         0,
         &denoiserParams,
         (CUdeviceptr)denoiserState,
         denoiserSizes.stateSizeInBytes,
         &guideLayer,
         &layer,
         1,
         0,
         0,
         (CUdeviceptr)denoiserScratch,
         denoiserSizes.withoutOverlapScratchSizeInBytes
         );
    }

    OptixDenoiser        denoiser = {};
    OptixDenoiserOptions denoiserOptions;
    void                *denoiserScratch = 0;
    void                *denoiserState   = 0;
    OptixDenoiserSizes   denoiserSizes;
    
    float4              *denoiserInput   = 0;
    float4              *denoiserOutput  = 0;
    float4              *denoiserNormal  = 0;
  };
#endif
#endif
      
  FrameBuffer::FrameBuffer(Context *context, const bool isOwner)
    : Object(context),
      isOwner(isOwner)
  {
    perDev.resize(context->devices.size());
    for (int localID=0;localID<context->devices.size();localID++) {
      perDev[localID]
        = TiledFB::create(context->getDevice(localID),this);
    }
  }

  FrameBuffer::~FrameBuffer()
  {
    freeResources();
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
    if (finalDepth) {
      BARNEY_CUDA_CALL(Free(finalDepth));
      finalDepth = 0;
    }
    if (finalColor) {
      BARNEY_CUDA_CALL(Free(finalColor));
      finalColor = 0;
    }
    denoiser = 0;
  }

  template<bool SRGB>
  __global__
  void toFixed8(uint32_t *out,
                float4 *in,
                vec2i numPixels)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    if (ix >= numPixels.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    if (iy >= numPixels.y) return;
    int idx = ix+numPixels.x*iy;
    
    float4 v = in[idx];
    v.x = clamp(v.x);
    v.y = clamp(v.y);
    v.z = clamp(v.z);
    if (SRGBA) { 
      v.x = powf(v.x,2.2f);
      v.y = powf(v.y,2.2f);
      v.z = powf(v.z,2.2f);
    }
    out[idx] = owl::make_rgba8(v);
  }

  void read(BNFrameBufferChannel channel,
            void *hostPtr,
            BNDataType requestedFormat)
  {
    if (!owner) return;

    if (dirty) {
      denoiser->run();
      dirty = false;
    }
    if (channel == BN_FB_DEPTH) {
      if (requestedFormat != BN_FLOAT)
        throw std::runtime_error("can only read depth channel as BN_FLOAT format");
      if (!finalDepth)
        throw std::runtime_error("requesting to read depth channel, but didn't create one");
      BARNEY_CUDA_CALL(Memcpy(hostPtr,finalDepth,
                              numPixels.x*numPixels.y*sizeof(float)));
      return;
    }

    if (channel != BN_FB_DEPTH)
      throw std::runtime_error("trying to read un-known channel!?");

    switch(requestedFormat) {
    case FB_FLOAT4_RGBA:
      BN_CUDA_CALL(MallocAsync((void**)&asFixed8,
                               numPixels.x*numPixel.y*sizeof(uint32_t),0));
      BARNEY_CUDA_CALL(Memcpy(hostPtr,finalDepth,
                              numPixels.x*numPixels.y*sizeof(float)));
      break;
    case FB_UFIXED8_RGBA: {
      uint32_t *asFixed8;
      BN_CUDA_CALL(MallocAsync((void**)&asFixed8,
                               numPixels.x*numPixel.y*sizeof(uint32_t),0));
      vec2i bs(8,8);
      toFixed8<false>
        <<<divRoundUp(numPixels,bs),bs>>>
        (asFixed8,finalColor,numPixels);
        BARNEY_CUDA_CALL(Memcpy(hostPtr,finalDepth,
                                numPixels.x*numPixels.y*sizeof(uint32_t)));
        BN_CUDA_CALL(FreeAsync(asFixed8,0));
    } break;
    case FB_UFIXED8_RGBA_SRGB: {
      uint32_t *asFixed8;
      BN_CUDA_CALL(MallocAsync((void**)&asFixed8,
                               numPixels.x*numPixel.y*sizeof(uint32_t),0));
      vec2i bs(8,8);
      toFixed8<false>
        <<<divRoundUp(numPixels,bs),bs>>>
        (asFixed8,finalColor,numPixels);
        BARNEY_CUDA_CALL(Memcpy(hostPtr,finalDepth,
                                numPixels.x*numPixels.y*sizeof(uint32_t)));
        BN_CUDA_CALL(FreeAsync(asFixed8,0));
    } break;
    default:
      throw std::runtime_error("requested to read color channel in un-supported format #"
                               +std::to_string((int)requestedFormat));
    };
  }
  
  void FrameBuffer::resize(vec2i size,
                           uint32_t channels)
  {
    // PING; PRINT(size);
    
    for (auto &pd: perDev)
      pd->resize(size);
    
    freeResources();
    numPixels = size;

    if (isOwner) {
      // allocate/resize a owner-only, device-side depth buffer that
      // we can write into in device kernels
      if (channels & BN_FB_DEPTH)
        // host wants a depth buffer, so we need to allocate one on
        // the device side for staging
        BARNEY_CUDA_CALL(Malloc(&finalDepth,
                                numPixels.x*numPixels.y*sizeof(float)));
      
      BARNEY_CUDA_CALL(Malloc(&finalColor, numPixels.x*numPixels.y*sizeof(float4)));
      
      if (!denoiser) denoiser = Denoiser::create(this);
      denoiser->resize();
    }
  }
    

}
