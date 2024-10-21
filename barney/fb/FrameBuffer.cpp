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

namespace barney {

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
    if (finalFB) {
      BARNEY_CUDA_CALL(Free(finalFB));
      finalFB = 0;
    }
  }

  void FrameBuffer::resize(vec2i size,
                           uint32_t *hostFB,
                           float    *hostDepth)
  {
    // PING; PRINT(size);
    
    for (auto &pd: perDev)
      pd->resize(size);
    
    freeResources();
    numPixels = size;

    if (isOwner) {
      // save the host pointers, which may be host-accesible only
      this->hostDepth = hostDepth;
      this->hostFB = hostFB;

      // allocate/resize a owner-only, device-side depth buffer that
      // we can write into in device kernels
      if (hostDepth)
        // host wants a depth buffer, so we need to allocate one on
        // the device side for staging
        BARNEY_CUDA_CALL(Malloc(&finalDepth,
                                numPixels.x*numPixels.y*sizeof(float)));

      PING; BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Malloc(&finalFB, numPixels.x*numPixels.y*sizeof(uint32_t)));


#if DENOISE
      PING; BARNEY_CUDA_SYNC_CHECK();
      if (!denoiserCreated) {
        // // if nonzero, albedo image must be given in OptixDenoiserGuideLayer
        // unsigned int guideAlbedo;
        denoiserOptions.guideAlbedo = 0;

        // // if nonzero, normal image must be given in OptixDenoiserGuideLayer
        // unsigned int guideNormal;
#if DENOISE_NORMAL
        denoiserOptions.guideNormal = 1;
#else
        denoiserOptions.guideNormal = 0;
#endif

        // /// alpha denoise mode
        // OptixDenoiserAlphaMode denoiseAlpha;
        // denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE;

        auto device = context->getDevice(0);

        OptixDeviceContext optixContext
          =
#if 1
          owlContextGetOptixContext(context->globalContextAcrossAllGPUs,0)
#else
          owlContextGetOptixContext(device->owl,0)
#endif
          ;
        PRINT(optixContext);
        PING; BARNEY_CUDA_SYNC_CHECK();
        optixDenoiserCreate(/*OptixDeviceContext */
                            optixContext,
                            /*OptixDenoiserModelKind*/
                            OPTIX_DENOISER_MODEL_KIND_HDR,
                            /*const OptixDenoiserOptions*/
                            &denoiserOptions,
                            /*OptixDenoiser*/
                            &denoiser);
        PING; PRINT(denoiser);
      }

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
      PING; BARNEY_CUDA_SYNC_CHECK();
      PRINT(denoiserSizes.overlapWindowSizeInPixels);
      if (denoiserScratch) BARNEY_CUDA_CALL(Free(denoiserScratch));
      PING; BARNEY_CUDA_SYNC_CHECK();
      denoiserScratch = 0;
      PING; BARNEY_CUDA_SYNC_CHECK();
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch,
                              denoiserSizes.withOverlapScratchSizeInBytes));
                              // denoiserSizes.withoutOverlapScratchSizeInBytes));
      PING; BARNEY_CUDA_SYNC_CHECK();

      if (denoiserState) BARNEY_CUDA_CALL(Free(denoiserState));
      BARNEY_CUDA_CALL(Malloc(&denoiserState,
                              denoiserSizes.stateSizeInBytes));
      PING; BARNEY_CUDA_SYNC_CHECK();
      denoiserState = 0;

      PING; BARNEY_CUDA_SYNC_CHECK();
      
      PRINT(numPixels);
      PRINT(denoiser);
      PRINT(denoiserSizes.stateSizeInBytes);
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
                         denoiserSizes.withOverlapScratchSizeInBytes
                         // denoiserSizes.withoutOverlapScratchSizeInBytes
                         );
      PING; BARNEY_CUDA_SYNC_CHECK();
      if (denoiserInput)
        BARNEY_CUDA_CALL(Free(denoiserInput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserInput,
                              numPixels.x*numPixels.y*sizeof(*denoiserInput)));
      PING; BARNEY_CUDA_SYNC_CHECK();
      if (denoiserOutput)
        BARNEY_CUDA_CALL(Free(denoiserOutput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(*denoiserOutput)));

      PING; BARNEY_CUDA_SYNC_CHECK();
# if DENOISE_NORMAL
      if (denoiserNormal)
        BARNEY_CUDA_CALL(Free(denoiserNormal));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserNormal,
                              numPixels.x*numPixels.y*sizeof(*denoiserNormal)));
# endif
#endif
      PING; BARNEY_CUDA_SYNC_CHECK();
    }
  }
    
#if DENOISE
  void FrameBuffer::denoise()
  {
    OptixDenoiserGuideLayer guideLayer = {};
    OptixDenoiserLayer layer = {};
    layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    layer.input.rowStrideInBytes = numPixels.x*sizeof(float4);
    layer.input.pixelStrideInBytes = sizeof(float4);
    layer.input.width = numPixels.x;
    layer.input.height = numPixels.y;
    layer.input.data = (CUdeviceptr)denoiserInput;

#if DENOISE_NORMAL
    guideLayer.normal = layer.input;
    guideLayer.normal.data = (CUdeviceptr)denoiserNormal;
#endif
    layer.output = layer.input;
    layer.output.data = (CUdeviceptr)denoiserOutput;

    OptixDenoiserParams denoiserParams = {};

    // PING; BARNEY_CUDA_SYNC_CHECK();
    optixDenoiserInvoke
      (
       // OptixDenoiser                   denoiser,
       denoiser,
       //          CUstream                        stream,
       0,
       //          const OptixDenoiserParams*      params,
       &denoiserParams,
       //          CUdeviceptr                     denoiserState,
       (CUdeviceptr)denoiserState,
       //          size_t                          denoiserStateSizeInBytes,
       denoiserSizes.stateSizeInBytes,
       //          const OptixDenoiserGuideLayer*  guideLayer,
       &guideLayer,
       //          const OptixDenoiserLayer*       layers,
       &layer,
       //          unsigned int                    numLayers,
       1,
       //          unsigned int                    inputOffsetX,
       0,
       //          unsigned int                    inputOffsetY,
       0,
       //          CUdeviceptr                     scratch,
       (CUdeviceptr)denoiserScratch,
       //          size_t                          scratchSizeInBytes );
       denoiserSizes.withoutOverlapScratchSizeInBytes
       );
    // PING; BARNEY_CUDA_SYNC_CHECK();
    float denoiseWeight = powf(.95f,std::max(0,(int)accumID-2));
    float4ToBGBA8(this->finalFB,
                  this->denoiserInput,
                  this->denoiserOutput,
                  denoiseWeight,
                  this->numPixels);
    // PING; BARNEY_CUDA_SYNC_CHECK();
  }
#endif

}
