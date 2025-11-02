// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
      // denoising alpha can get really funky results when using
      // compositing (eg in paraview/ice-t), so let's not do that.
      denoiserOptions.denoiseAlpha
        = OPTIX_DENOISER_ALPHA_MODE_COPY;
        
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
      if (in_rgba) {
        BARNEY_CUDA_CALL_NOTHROW(Free(in_rgba));
        in_rgba = 0;
      }
      if (in_normal) {
        BARNEY_CUDA_CALL_NOTHROW(Free(in_normal));
        in_normal = 0;
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
      // --------------------------------------------
      if (denoiserScratch) {
        BARNEY_CUDA_CALL(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch,
                              denoiserSizes.withoutOverlapScratchSizeInBytes));
      
      // --------------------------------------------
      if (denoiserState) {
        BARNEY_CUDA_CALL(Free(denoiserState));
        denoiserState = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserState,
                              denoiserSizes.stateSizeInBytes));
      // --------------------------------------------
      if (in_rgba) {
        BARNEY_CUDA_CALL(Free(in_rgba));
        in_rgba = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&in_rgba,
                              numPixels.x*numPixels.y*sizeof(*in_rgba)));
      // --------------------------------------------
      if (out_rgba) {
        BARNEY_CUDA_CALL(Free(out_rgba));
        out_rgba = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&out_rgba,
                              numPixels.x*numPixels.y*sizeof(*out_rgba)));
      // --------------------------------------------
      if (in_normal) {
        BARNEY_CUDA_CALL(Free(in_normal));
        in_normal = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&in_normal,
                              numPixels.x*numPixels.y*sizeof(*in_normal)));
      // --------------------------------------------
      
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
    
    void Optix8Denoiser::run(float blendFactor)
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

      /// blend factor.
      /// If set to 0 the output is 100% of the denoised input. If set to 1, the output is 100% of
      /// the unmodified input. Values between 0 and 1 will linearly interpolate between the denoised
      /// and unmodified input.
      denoiserParams.blendFactor      = blendFactor;
      // iw - this should at some point use the stream used for rendering/copy pixels
      cudaStream_t denoiserStream = 0;
      optixDenoiserInvoke
        (
         denoiser,
         denoiserStream,
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
      cudaStreamSynchronize(denoiserStream);
    }
    
#endif
    
  }
}

