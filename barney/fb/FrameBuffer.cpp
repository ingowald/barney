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

      BARNEY_CUDA_CALL(Malloc(&finalFB, numPixels.x*numPixels.y*sizeof(uint32_t)));


#if DENOISE
      if (!denoiserCreated) {
        // // if nonzero, albedo image must be given in OptixDenoiserGuideLayer
        // unsigned int guideAlbedo;
        denoiserOptions.guideAlbedo = 0;

        // // if nonzero, normal image must be given in OptixDenoiserGuideLayer
        // unsigned int guideNormal;
        denoiserOptions.guideNormal = 0;

        // /// alpha denoise mode
        // OptixDenoiserAlphaMode denoiseAlpha;
        // denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE;

        auto device = context->getDevice(0);

        OptixDeviceContext optixContext
          = owlContextGetOptixContext(context->globalContextAcrossAllGPUs,0);//device->devGroup->owl,device->owlID);
        optixDenoiserCreate(/*OptixDeviceContext */
                            optixContext,
                            /*OptixDenoiserModelKind*/
                            // OPTIX_DENOISER_MODEL_KIND_LDR,
                            OPTIX_DENOISER_MODEL_KIND_HDR,
                            /*const OptixDenoiserOptions*/
                            &denoiserOptions,
                            /*OptixDenoiser*/
                            &denoiser);
      }

      optixDenoiserComputeMemoryResources(/*const OptixDenoiser */
                                          denoiser,
                                          // unsigned int        outputWidth,
                                          numPixels.x,
                                          // unsigned int        outputHeight,
                                          numPixels.y,
                                          // OptixDenoiserSizes* returnSizes
                                          &denoiserSizes
                                          );
      if (denoiserScratch) BARNEY_CUDA_CALL(Free(denoiserScratch));
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch,
                              denoiserSizes.withoutOverlapScratchSizeInBytes));

      if (denoiserState) BARNEY_CUDA_CALL(Free(denoiserState));
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
      if (denoiserInput)
        BARNEY_CUDA_CALL(Free(denoiserInput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserInput,
                              numPixels.x*numPixels.y*sizeof(*denoiserInput)));
      if (denoiserOutput)
        BARNEY_CUDA_CALL(Free(denoiserOutput));
      BARNEY_CUDA_CALL(Malloc((void **)&denoiserOutput,
                              numPixels.x*numPixels.y*sizeof(*denoiserOutput)));
#endif
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
      
    layer.output = layer.input;
    layer.output.data = (CUdeviceptr)denoiserOutput;

    OptixDenoiserParams denoiserParams = {};
      
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
    float4ToBGBA8(this->finalFB,this->denoiserOutput,this->numPixels);
  }
#endif

}
