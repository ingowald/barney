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
#include <cuda_runtime.h>
#if BARNEY_HAVE_OIDN
# include <OpenImageDenoise/oidn.h>
#endif

namespace barney {

  inline __device__ float saturate(float f, float lo=0.f, float hi=1.f)
  { return max(lo,min(f,hi)); }
  
  inline __device__ float from_8bit(uint8_t v) {
    return float(v) * (1.f/255.f);
  }
  
  inline __device__ vec4f from_8bit(uint32_t v) {
    return vec4f(from_8bit(uint8_t((v >> 0)&0xff)),
                 from_8bit(uint8_t((v >> 8)&0xff)),
                 from_8bit(uint8_t((v >> 16)&0xff)),
                 from_8bit(uint8_t((v >> 24)&0xff)));
  }
  
  inline __device__ float linear_to_srgb(float x) {
    if (x <= 0.0031308f) {
      return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
  }

  inline __device__ uint32_t _make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __device__ uint32_t make_rgba8(const vec4f color, bool dbg=false)
  {
    if (dbg)
      printf("col %f %f %f %f\n",
             color.x,
             color.y,
             color.z,
             color.w);
    uint32_t r = _make_8bit(color.x);
    uint32_t g = _make_8bit(color.y);
    uint32_t b = _make_8bit(color.z);
    uint32_t a = 0xff; //make_8bit(color.w);
    uint32_t ret =
      (r << 0) |
      (g << 8) |
      (b << 16) |
      (a << 24);
    // if (dbg) printf("%x %x %x %x all %x\n",
    //                 r,g,b,a,ret);
    return ret;
      // (_make_8bit(color.x) << 0) +
      // (_make_8bit(color.y) << 8) +
      // (_make_8bit(color.z) << 16) +
      // (_make_8bit(color.w) << 24);
  }
  
  inline __device__ float clamp(float f) { return min(1.f,max(0.f,f)); }

  __global__ void copyPixels(vec2i numPixels,
                             float4 *out,
                             vec3f *in_color,
                             float *in_alpha)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    if (ix >= numPixels.x) return;
    if (iy >= numPixels.y) return;
    int idx = ix + numPixels.x*iy;
    vec3f color = in_color[idx];
    float alpha = in_alpha[idx];
    out[idx] = vec4f(color,alpha);
  }
  
  struct DenoiserNone : public Denoiser {
    DenoiserNone(FrameBuffer *fb) : Denoiser(fb) {};
    virtual ~DenoiserNone() {}
    void resize() override { }
                                      
    void run() override
    {
      vec2i bs(8,8);
      CHECK_CUDA_LAUNCH(copyPixels,
                        divRoundUp(fb->numPixels,bs),bs,0,0,
                        //
                        fb->numPixels,fb->denoisedColor,
                        fb->linearColor,fb->linearAlpha);
      BARNEY_CUDA_SYNC_CHECK();
    }
  };

#if !BARNEY_DISABLE_DENOISING
#if BARNEY_HAVE_OIDN
  struct DenoiserOIDN : public Denoiser {
    DenoiserOIDN(FrameBuffer *fb)
      : Denoiser(fb)
    {
      int devID = 0;
      cudaStream_t stream = 0;
      device = 
        oidnNewCUDADevice(&devID,&stream,1);
      oidnCommitDevice(device);
      filter = oidnNewFilter(device,"RT");
    }
    virtual ~DenoiserOIDN()
    {
      if (colorBuf)  oidnReleaseBuffer(colorBuf);
      if (normalBuf) oidnReleaseBuffer(normalBuf);
      if (outputBuf) oidnReleaseBuffer(outputBuf);
      if (denoiserOutput) BARNEY_CUDA_CALL_NOTHROW(Free(denoiserOutput));
      oidnReleaseFilter(filter);
      oidnReleaseDevice(device);
    }
    
    void resize() override
    {
      vec2i numPixels = fb->numPixels;
      if (denoiserOutput)
        BARNEY_CUDA_CALL(Free(denoiserOutput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(float3)));
      // if (fb->linearColor)
      //   BARNEY_CUDA_CALL(Free(denoiserInput));
      // BARNEY_CUDA_CALL(Malloc((void **)&denoiserInput,
      //                         numPixels.x*numPixels.y*sizeof(*denoiserInput)));
    
      // if (denoiserAlpha)
      //   BARNEY_CUDA_CALL(Free(denoiserAlpha));
      // BARNEY_CUDA_CALL(Malloc((void **)&denoiserAlpha,
      //                         numPixels.x*numPixels.y*sizeof(*denoiserAlpha)));
      // if (denoiserOutput)
      //   BARNEY_CUDA_CALL(Free(denoiserOutput));
      // BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
      //                         numPixels.x*numPixels.y*sizeof(*denoiserOutput)));
      
      // if (denoiserNormal)
      //   BARNEY_CUDA_CALL(Free(denoiserNormal));

      // BARNEY_CUDA_CALL(Malloc((void **)&denoiserNormal,
      //                         numPixels.x*numPixels.y*sizeof(*denoiserNormal)));
      if (colorBuf)  oidnReleaseBuffer(colorBuf);
      if (normalBuf) oidnReleaseBuffer(normalBuf);
      if (outputBuf) oidnReleaseBuffer(outputBuf);
      colorBuf
        = oidnNewSharedBuffer(device, fb->linearColor,
                              numPixels.x*numPixels.y*sizeof(*fb->linearColor));
      normalBuf
        = oidnNewSharedBuffer(device, fb->linearNormal,
                              numPixels.x*numPixels.y*sizeof(*fb->linearNormal));
      outputBuf
        = oidnNewSharedBuffer(device, denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(*denoiserOutput));
      oidnSetFilterImage(filter,"color",colorBuf,
                         OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      // oidnSetFilterImage(filter,"normal",normalBuf,
      //                    OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      // oidnSetFilterImage(filter,"albedo",normalBuf,
                         // OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      oidnSetFilterImage(filter,"output",outputBuf,
                         OIDN_FORMAT_FLOAT3,numPixels.x,numPixels.y,0,0,0);
      oidnSetFilterBool(filter,"hdr",true);
      oidnCommitFilter(filter);
    }
    void run() override
    {
      oidnExecuteFilter(filter);
      vec2i bs(8,8);
      copyPixels<<<divRoundUp(fb->numPixels,bs),bs>>>
        (fb->numPixels,fb->denoisedColor,denoiserOutput,fb->linearAlpha);
      const char *error;
      oidnGetDeviceError(device,&error);
      if (error)
        PRINT(error);
    }
    
    vec3f    *denoiserOutput   = 0;
    
    OIDNBuffer outputBuf = 0; 
    OIDNBuffer normalBuf = 0; 
    OIDNBuffer colorBuf = 0; 
    OIDNDevice device = 0;
    OIDNFilter filter = 0;
  };
#endif


  __global__ void float3_to_float4(float4 *out,
                                   float3 *in,
                                   int count)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= count) return;
    float3 v = in[tid];
    out[tid] = make_float4(v.x,v.y,v.z,0.f);
  }
  
#if OPTIX_VERSION >= 80000
  struct DenoiserOptix : public Denoiser {
    DenoiserOptix(FrameBuffer *fb)
      : Denoiser(fb)
    {
      denoiserOptions.guideAlbedo = 0;
      denoiserOptions.guideNormal = 0;//1;
      denoiserOptions.denoiseAlpha
        = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
        
      auto device = fb->context->getDevice(0);

      OptixDeviceContext optixContext
        = owlContextGetOptixContext(device->devGroup->owl,0);
      optixDenoiserCreate(optixContext,
                          OPTIX_DENOISER_MODEL_KIND_HDR,
                          &denoiserOptions,
                          &denoiser);
    }      
    virtual ~DenoiserOptix() {
      if (denoiserNormal)
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserNormal));
      if (denoiserInput)
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserInput));
      if (denoiserScratch) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      if (denoiserState) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserState));
        denoiserState = 0;
      }
    }
    void resize() override
    {
      vec2i numPixels = fb->numPixels;
      if (denoiserInput)
        BARNEY_CUDA_CALL(Free(denoiserInput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserInput,
                              numPixels.x*numPixels.y*sizeof(*denoiserInput)));
    
      // if (denoiserOutput)
      //   BARNEY_CUDA_CALL(Free(denoiserOutput));
      // BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
      //                         numPixels.x*numPixels.y*sizeof(*denoiserOutput)));
      
      if (denoiserNormal)
        BARNEY_CUDA_CALL(Free(denoiserNormal));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserNormal,
                              numPixels.x*numPixels.y*sizeof(*denoiserNormal)));
      denoiserSizes.overlapWindowSizeInPixels = 0;
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
      auto numPixels = fb->numPixels;
      layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT3;
      layer.input.rowStrideInBytes = numPixels.x*sizeof(float4);
      layer.input.pixelStrideInBytes = sizeof(float4);
      layer.input.width = numPixels.x;
      layer.input.height = numPixels.y;
      float3_to_float4
        <<<divRoundUp(numPixels.x*numPixels.y,1024),1024>>>
        (denoiserInput,(float3*)fb->linearColor,numPixels.x*numPixels.y);
      layer.input.data = (CUdeviceptr)denoiserInput;
      // layer.input.data = (CUdeviceptr)fb->linearColor;//denoiserInput;

      float3_to_float4
        <<<divRoundUp(numPixels.x*numPixels.y,1024),1024>>>
        (denoiserNormal,(float3*)fb->linearNormal,numPixels.x*numPixels.y);
      guideLayer.normal = layer.input;
      guideLayer.normal.data = (CUdeviceptr)denoiserNormal;
      layer.output = layer.input;
      layer.output.data = (CUdeviceptr)fb->denoisedColor;//denoiserOutput;

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
      BARNEY_CUDA_SYNC_CHECK();
    }

    OptixDenoiser        denoiser = {};
    OptixDenoiserOptions denoiserOptions;
    void                *denoiserScratch = 0;
    void                *denoiserState   = 0;
    OptixDenoiserSizes   denoiserSizes;
    
    float4              *denoiserInput   = 0;
    float4              *denoiserNormal   = 0;
    // float4              *denoiserOutput  = 0;
    // float3              *denoiserOutput  = 0;
    // float4              *denoiserNormal  = 0;
  };
#endif
#endif

  Denoiser::SP Denoiser::create(FrameBuffer *fb)
  {
#if BARNEY_DISABLE_DENOISING
    return std::make_shared<DenoiserNone>(fb);
#else
# if BARNEY_HAVE_OIDN
    return std::make_shared<DenoiserOIDN>(fb);
# endif
# if OPTIX_VERSION >= 80000
    return std::make_shared<DenoiserOptix>(fb);
# else
    return std::make_shared<DenoiserNone>(fb);
# endif
#endif
  }
  
  
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
    if (denoisedColor) {
      BARNEY_CUDA_CALL(Free(denoisedColor));
      denoisedColor = 0;
    }
    if (linearColor) {
      BARNEY_CUDA_CALL(Free(linearColor));
      linearColor = 0;
    }
    if (linearAlpha) {
      BARNEY_CUDA_CALL(Free(linearAlpha));
      linearAlpha = 0;
    }
    if (linearDepth) {
      BARNEY_CUDA_CALL(Free(linearDepth));
      linearDepth = 0;
    }
    if (linearNormal) {
      BARNEY_CUDA_CALL(Free(linearNormal));
      linearNormal = 0;
    }
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

    bool dbg = 0; //(ix == 0 && iy == 0);
    
    float4 v = in[idx];
    if (dbg) printf("tofixed in  %f %f %f %f\n",v.x,v.y,v.z,v.w);
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
    out[idx] = make_rgba8(v,dbg);
  }

  __global__ void toneMap(float4 *color, vec2i numPixels)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    if (ix >= numPixels.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    if (iy >= numPixels.y) return;
    int idx = ix+numPixels.x*iy;

    float4 v = color[idx];
#if 1
    v.x = linear_to_srgb(v.x);
    v.y = linear_to_srgb(v.y);
    v.z = linear_to_srgb(v.z);
#elif 0
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


  void FrameBuffer::finalizeFrame()
  {
    dirty = true;
    ownerGatherCompressedTiles();
    if (isOwner) {
      unpackTiles();
    }
    
  }


  __global__ void g_unpackTiles(vec2i numPixels,
                                vec3f *colors,
                                float *alphas,
                                vec3f *normals,
                                float *depths,
                                CompressedTile *tiles,
                                TileDesc *descs)
  {
    int tileIdx = blockIdx.x;

    const CompressedTile &tile = tiles[tileIdx];
    const TileDesc        desc = descs[tileIdx];
    
    int subIdx = threadIdx.x;
    int iix = subIdx % tileSize;
    int iiy = subIdx / tileSize;
    int ix = desc.lower.x + iix;
    int iy = desc.lower.y + iiy;
    if (ix >= numPixels.x) return;
    if (iy >= numPixels.y) return;
    int idx = ix + numPixels.x*iy;
    
    vec4f rgba = from_8bit(tile.rgba[subIdx]);
    float alpha = rgba.w;
    float scale = float(tile.scale[subIdx]);
    vec3f color = vec3f(rgba.x,rgba.y,rgba.z)*scale;
    // if (ix == 0 && iy == 0)
    //   printf("rgba %f %f %f scale %f color %f %f %f\n",
    //          rgba.x,rgba.y,rgba.z,scale,
    //          color.x,color.y,color.z);
    vec3f normal = tile.normal[subIdx].get3f();
    float depth = tile.depth[subIdx];

    colors[idx] = color;
    alphas[idx] = alpha;
    depths[idx] = depth;
    normals[idx] = normal;
  }
  
  void FrameBuffer::unpackTiles()
  {
#if 1
    CHECK_CUDA_LAUNCH(g_unpackTiles,
                      //
                      gatheredTilesOnOwner.numActiveTiles,pixelsPerTile,0,0,
                      //
                      numPixels,
                      linearColor,
                      linearAlpha,
                      linearNormal,
                      linearDepth,
                      gatheredTilesOnOwner.compressedTiles,
                      gatheredTilesOnOwner.tileDescs);
#else
    g_unpackTiles<<<gatheredTilesOnOwner.numActiveTiles,pixelsPerTile>>>
      (numPixels,
       linearColor,
       linearAlpha,
       linearNormal,
       linearDepth,
       gatheredTilesOnOwner.compressedTiles,
       gatheredTilesOnOwner.tileDescs);
#endif
  }

  void FrameBuffer::read(BNFrameBufferChannel channel,
                         void *hostPtr,
                         BNDataType requestedFormat)
  {
    if (!isOwner) return;

    if (dirty) {
      denoiser->run();
      vec2i bs(8,8);
      CHECK_CUDA_LAUNCH(toneMap,
                        divRoundUp(numPixels,bs),bs,0,0,
                        //
                        denoisedColor,numPixels);
      // toneMap<<<divRoundUp(numPixels,bs),bs>>>(denoisedColor,numPixels);
      BARNEY_CUDA_SYNC_CHECK();
      dirty = false;
    }
    if (channel == BN_FB_DEPTH && hostPtr && linearDepth) {
      if (requestedFormat != BN_FLOAT)
        throw std::runtime_error("can only read depth channel as BN_FLOAT format");
      if (!linearDepth)
        throw std::runtime_error("requesting to read depth channel, but didn't create one");
      BARNEY_CUDA_CALL(Memcpy(hostPtr,linearDepth,
                              numPixels.x*numPixels.y*sizeof(float),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      return;
    }

    if (!hostPtr) return;
    
    if (channel != BN_FB_COLOR)
      throw std::runtime_error("trying to read un-known channel!?");

    BARNEY_CUDA_SYNC_CHECK();
    
    switch(requestedFormat) {
    case BN_FLOAT4: 
    case BN_FLOAT4_RGBA: {
      BARNEY_CUDA_CALL(Memcpy(hostPtr,denoisedColor,
                              numPixels.x*numPixels.y*sizeof(float4),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
    } break;
    case BN_UFIXED8_RGBA: {
      uint32_t *asFixed8;
      BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Malloc((void**)&asFixed8,
                              numPixels.x*numPixels.y*sizeof(uint32_t)));
      BARNEY_CUDA_SYNC_CHECK();
      vec2i bs(8,8);
      CHECK_CUDA_LAUNCH(toFixed8<false>,
                        divRoundUp(numPixels,bs),bs,0,0,
                        asFixed8,denoisedColor,numPixels);
      BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Memcpy(hostPtr,asFixed8,
                              numPixels.x*numPixels.y*sizeof(uint32_t),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Free(asFixed8));
    } break;
    case BN_UFIXED8_RGBA_SRGB: {
      uint32_t *asFixed8;
      BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Malloc((void**)&asFixed8,
                                   numPixels.x*numPixels.y*sizeof(uint32_t)));
      BARNEY_CUDA_SYNC_CHECK();
      vec2i bs(8,8);
      CHECK_CUDA_LAUNCH(toFixed8<true>,
                        divRoundUp(numPixels,bs),bs,0,0,
                        asFixed8,denoisedColor,numPixels);
      BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Memcpy(hostPtr,asFixed8,
                              numPixels.x*numPixels.y*sizeof(uint32_t),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Free(asFixed8));
    } break;
    default:
      throw std::runtime_error("requested to read color channel in un-supported format #"
                               +std::to_string((int)requestedFormat));
    };
  }
  
  void FrameBuffer::resize(vec2i size,
                           uint32_t channels)
  {
    for (auto &pd: perDev)
      pd->resize(size);
    
    freeResources();
    numPixels = size;

    if (isOwner) {
      BARNEY_CUDA_CALL(Malloc(&denoisedColor,
                              numPixels.x*numPixels.y*sizeof(*denoisedColor)));
      BARNEY_CUDA_CALL(Malloc(&linearDepth,
                              numPixels.x*numPixels.y*sizeof(*linearDepth)));
      BARNEY_CUDA_CALL(Malloc(&linearColor,
                              numPixels.x*numPixels.y*sizeof(*linearColor)));
      BARNEY_CUDA_CALL(Malloc(&linearAlpha,
                              numPixels.x*numPixels.y*sizeof(*linearAlpha)));
      BARNEY_CUDA_CALL(Malloc(&linearNormal,
                              numPixels.x*numPixels.y*sizeof(*linearNormal)));
      
      if (!denoiser) denoiser = Denoiser::create(this);
      denoiser->resize();
    }
  }
    

}
