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

#include "rtcore/optix/Denoiser.h"
#include <optix.h>
// #include <optix_function_table.h>
#include <optix_stubs.h>

namespace rtc {
  namespace optix {
    
#if OPTIX_VERSION >= 80000
    
    Optix8Denoiser::Optix8Denoiser(Device *device)
      : Denoiser(device)
    {
      SetActiveGPU forDuration(device);
      denoiserOptions.guideAlbedo = 0;
      denoiserOptions.guideNormal = 1;
      denoiserOptions.denoiseAlpha
        = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
        
      OptixDeviceContext optixContext
        = owlContextGetOptixContext(device->owl,0);
      optixDenoiserCreate(optixContext,
                          OPTIX_DENOISER_MODEL_KIND_HDR,
                          &denoiserOptions,
                          &denoiser);
    }
    
    Optix8Denoiser::~Optix8Denoiser()
    {
      SetActiveGPU forDuration(device);
      if (denoiserScratch) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      if (denoiserState) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserState));
        denoiserState = 0;
      }
    }
    
    void Optix8Denoiser::resize(vec2i numPixels)
    {
      this->numPixels = numPixels;
      SetActiveGPU forDuration(device);
    
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
    
    void Optix8Denoiser::run(// output
                       vec4f *out_rgba,
                       // input channels
                       vec4f *in_rgba,
                       vec3f *in_normal,
                       float blendFactor)
    {
      SetActiveGPU forDuration(device);
      OptixDenoiserLayer layer = {};
      
      layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
      layer.input.rowStrideInBytes = numPixels.x*sizeof(vec4f);
      layer.input.pixelStrideInBytes = sizeof(vec4f);
      layer.input.width  = numPixels.x;
      layer.input.height = numPixels.y;
      layer.input.data   = (CUdeviceptr)in_rgba;
      
      OptixDenoiserGuideLayer guideLayer = {};
      guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
      guideLayer.normal.rowStrideInBytes = numPixels.x*sizeof(vec3f);
      guideLayer.normal.pixelStrideInBytes = sizeof(vec3f);
      guideLayer.normal.width  = numPixels.x;
      guideLayer.normal.height = numPixels.y;
      guideLayer.normal.data = (CUdeviceptr)in_normal;
      
      layer.output = layer.input;
      layer.output.data = (CUdeviceptr)out_rgba;

      OptixDenoiserParams denoiserParams = {};
      denoiserParams.blendFactor      = blendFactor;

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
    
#endif
    
  }
}

