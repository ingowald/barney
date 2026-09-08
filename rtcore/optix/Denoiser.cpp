// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/optix/Denoiser.h"
#include <optix.h>
// #include <optix_function_table.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>
#include <algorithm>
#include <cstdint>
#include <iostream>

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

      currentUpscaleMode = upscaleMode;
      OptixDenoiserModelKind modelKind
        = upscaleMode
        ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X
        : OPTIX_DENOISER_MODEL_KIND_AOV;

      OptixDeviceContext optixContext
        = owlContextGetOptixContext(device->owl,0);
      OptixResult res = optixDenoiserCreate(optixContext,
                          modelKind,
                          &denoiserOptions,
                          &denoiser);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): OptiX denoiser creation failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        denoiser = {};
        available = false;
      }
    }

    void Optix8Denoiser::recreateIfNeeded()
    {
      if (!available) return;
      if (currentUpscaleMode == upscaleMode) return;

      SetActiveGPU forDuration(device);

      // destroy the existing OptixDenoiser handle
      if (denoiser) {
        optixDenoiserDestroy(denoiser);
        denoiser = {};
      }

      currentUpscaleMode = upscaleMode;
      OptixDenoiserModelKind modelKind
        = upscaleMode
        ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X
        : OPTIX_DENOISER_MODEL_KIND_AOV;

      OptixDeviceContext optixContext
        = owlContextGetOptixContext(device->owl,0);
      OptixResult res = optixDenoiserCreate(optixContext,
                          modelKind,
                          &denoiserOptions,
                          &denoiser);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): OptiX denoiser re-creation failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        denoiser = {};
        available = false;
      }
    }
    
    Optix8Denoiser::~Optix8Denoiser()
    {
      SetActiveGPU forDuration(device);
      if (denoiser) {
        optixDenoiserDestroy(denoiser);
        denoiser = {};
      }
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
      if (out_rgba) {
        BARNEY_CUDA_CALL_NOTHROW(Free(out_rgba));
        out_rgba = 0;
      }
      if (in_normal) {
        BARNEY_CUDA_CALL_NOTHROW(Free(in_normal));
        in_normal = 0;
      }
      if (denoiserIntensity) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserIntensity));
        denoiserIntensity = 0;
      }
      if (denoiserAvgColor) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserAvgColor));
        denoiserAvgColor = 0;
      }
    }
    
    void Optix8Denoiser::resize(vec2i numPixels)
    {
      // If upscale mode changed, destroy and recreate the denoiser
      recreateIfNeeded();

      this->numPixels = numPixels;
      // output is 2x input when upscaling, same as input otherwise
      outputDims = upscaleMode
        ? vec2i(numPixels.x*2, numPixels.y*2)
        : numPixels;

      if (!available) return;

      SetActiveGPU forDuration(device);

      denoiserSizes.overlapWindowSizeInPixels = 0;
      // Clamp denoise resolution to a tile so the uint32 tensor element count
      // can't overflow on large frames; run() tiles via InvokeTiled. UPSCALE2X
      // sizes its tensor from the 2x output, so halve the input tile there.
      constexpr uint32_t kMaxTile = 2048;
      const uint32_t maxTile = upscaleMode ? kMaxTile / 2 : kMaxTile;
      tileDims.x = std::min<uint32_t>(maxTile, (unsigned int)numPixels.x);
      tileDims.y = std::min<uint32_t>(maxTile, (unsigned int)numPixels.y);
      OptixResult res = optixDenoiserComputeMemoryResources(denoiser, tileDims.x, tileDims.y, &denoiserSizes);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): optixDenoiserComputeMemoryResources failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
      overlapWindow = denoiserSizes.overlapWindowSizeInPixels;
      // --------------------------------------------
      if (denoiserScratch) {
        BARNEY_CUDA_CALL(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      // tiled invoke needs the with-overlap scratch
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch,
                              denoiserSizes.withOverlapScratchSizeInBytes));
      
      // --------------------------------------------
      if (denoiserState) {
        BARNEY_CUDA_CALL(Free(denoiserState));
        denoiserState = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserState,
                              denoiserSizes.stateSizeInBytes));
      // --- input buffers at render resolution ---
      if (in_rgba) {
        BARNEY_CUDA_CALL(Free(in_rgba));
        in_rgba = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&in_rgba,
                              (size_t)numPixels.x*numPixels.y*sizeof(*in_rgba)));
      // --- output buffer at output (possibly 2x) resolution ---
      if (out_rgba) {
        BARNEY_CUDA_CALL(Free(out_rgba));
        out_rgba = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&out_rgba,
                              (size_t)outputDims.x*outputDims.y*sizeof(*out_rgba)));
      // --- normal guide at render resolution ---
      if (in_normal) {
        BARNEY_CUDA_CALL(Free(in_normal));
        in_normal = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&in_normal,
                              (size_t)numPixels.x*numPixels.y*sizeof(*in_normal)));
      // --- whole-frame autoexposure intensity (single float) ---
      if (!denoiserIntensity)
        BARNEY_CUDA_CALL(Malloc(&denoiserIntensity, sizeof(*denoiserIntensity)));
      // --- whole-frame average log color (three floats) ---
      constexpr size_t kAvgColorChannels = 3;
      if (!denoiserAvgColor)
        BARNEY_CUDA_CALL(Malloc(&denoiserAvgColor,
                                kAvgColorChannels*sizeof(*denoiserAvgColor)));
      // --------------------------------------------
      
      // Setup takes INPUT dims; UPSCALE2X produces 2x output from them (passing
      // output dims would yield 4x). Size to the largest padded tile fed by the
      // tiled invoke (tile + 2*overlap).
      res = optixDenoiserSetup(denoiser,
                         0,
                         tileDims.x + 2 * overlapWindow,
                         tileDims.y + 2 * overlapWindow,
                         (CUdeviceptr)denoiserState,
                         denoiserSizes.stateSizeInBytes,
                         (CUdeviceptr)denoiserScratch,
                         denoiserSizes.withOverlapScratchSizeInBytes
                         );
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): optixDenoiserSetup failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
    }
    
    void Optix8Denoiser::run(float blendFactor)
    {
      if (!available) return;
      SetActiveGPU forDuration(device);
      OptixDenoiserLayer layer = {};
      
      // --- input at render resolution ---
      layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
      layer.input.rowStrideInBytes = numPixels.x*sizeof(vec4f);
      layer.input.pixelStrideInBytes = sizeof(vec4f);
      layer.input.width  = numPixels.x;
      layer.input.height = numPixels.y;
      layer.input.data   = (CUdeviceptr)in_rgba;
      
      // --- normal guide at render resolution ---
      OptixDenoiserGuideLayer guideLayer = {};
      guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
      guideLayer.normal.rowStrideInBytes = numPixels.x*sizeof(vec3f);
      guideLayer.normal.pixelStrideInBytes = sizeof(vec3f);
      guideLayer.normal.width  = numPixels.x;
      guideLayer.normal.height = numPixels.y;
      guideLayer.normal.data = (CUdeviceptr)in_normal;
      
      // --- output at outputDims (render res or 2x when upscaling) ---
      layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;
      layer.output.rowStrideInBytes = outputDims.x*sizeof(vec4f);
      layer.output.pixelStrideInBytes = sizeof(vec4f);
      layer.output.width  = outputDims.x;
      layer.output.height = outputDims.y;
      layer.output.data = (CUdeviceptr)out_rgba;

      OptixDenoiserParams denoiserParams = {};

      /// blend factor.
      /// If set to 0 the output is 100% of the denoised input. If set to 1, the output is 100% of
      /// the unmodified input. Values between 0 and 1 will linearly interpolate between the denoised
      /// and unmodified input.
      denoiserParams.blendFactor      = blendFactor;
      cudaStream_t denoiserStream = 0;

      // Whole-frame autoexposure so tiles share one intensity; per-tile
      // normalization would jump brightness at seams.
      OptixResult res = optixDenoiserComputeIntensity
        (denoiser, denoiserStream, &layer.input,
         (CUdeviceptr)denoiserIntensity,
         (CUdeviceptr)denoiserScratch,
         denoiserSizes.withOverlapScratchSizeInBytes);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): optixDenoiserComputeIntensity failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
      denoiserParams.hdrIntensity = (CUdeviceptr)denoiserIntensity;

      // Same reasoning for the AOV model's average log color: left null it is
      // auto-computed per tile, which makes tiles disagree on color balance.
      res = optixDenoiserComputeAverageColor
        (denoiser, denoiserStream, &layer.input,
         (CUdeviceptr)denoiserAvgColor,
         (CUdeviceptr)denoiserScratch,
         denoiserSizes.withOverlapScratchSizeInBytes);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): optixDenoiserComputeAverageColor failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
      denoiserParams.hdrAverageColor = (CUdeviceptr)denoiserAvgColor;

      // Descriptors stay full-frame; the helper slices them per tile (single
      // invoke when the frame fits one tile).
      res = optixUtilDenoiserInvokeTiled
        (
         denoiser,
         denoiserStream,
         &denoiserParams,
         (CUdeviceptr)denoiserState,
         denoiserSizes.stateSizeInBytes,
         &guideLayer,
         &layer,
         1,
         (CUdeviceptr)denoiserScratch,
         denoiserSizes.withOverlapScratchSizeInBytes,
         overlapWindow,
         tileDims.x,
         tileDims.y
         );
      if (res != OPTIX_SUCCESS) {
        std::cerr << "#barney(warn): optixUtilDenoiserInvokeTiled failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
      cudaStreamSynchronize(denoiserStream);
    }
    
#endif
    
  }
}

