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
#include "rtcore/optix/DenoiserUtils.h"
#include "barney/common/DenoiserConfig.h"
#include <optix.h>
// #include <optix_function_table.h>
#include <optix_stubs.h>

// For advanced pixel conversion (if thrust is available)
#ifdef BARNEY_HAVE_THRUST
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#endif

// For performance instrumentation
#include <chrono>

// Simple instrumentation utility inspired by RTX denoiser
namespace {
  struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    const char* name;
    
    Timer(const char* n) : name(n) {
      start = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      // Only print timing if environment variable is set
      if (getenv("BARNEY_DENOISER_TIMING")) {
        printf("Denoiser %s: %ld µs\n", name, duration.count());
      }
    }
  };
}

namespace rtc {
  namespace optix {
    
#if OPTIX_VERSION >= 80000
    
    Optix8Denoiser::Optix8Denoiser(Device *device)
      : Denoiser(device)
    {
      init();
    }
    
    void Optix8Denoiser::init()
    {
      if (denoiser)
        return;
        
      SetActiveGPU forDuration(device);
      
      // Configure OptiX denoiser options using centralized configuration
      // These options determine which guide layers are required and how alpha is handled
      
      // Guide layers provide additional information to improve denoising quality:
      // - Albedo guide: Surface color without lighting (improves material preservation)
      // - Normal guide: Surface normals (improves geometric detail preservation)
      // Enabling guide layers increases memory usage but typically improves quality
      denoiserOptions.guideAlbedo = BARNEY_NS::denoiser::optix::GUIDE_ALBEDO_DEFAULT;
      denoiserOptions.guideNormal = BARNEY_NS::denoiser::optix::GUIDE_NORMAL_DEFAULT;
      
      // Alpha channel handling options:
      // - COPY: Preserve original alpha values (faster, but alpha may remain noisy)  
      // - DENOISE: Apply denoising to alpha channel (slower, but cleaner transparency)
      denoiserOptions.denoiseAlpha = static_cast<OptixDenoiserAlphaMode>(
          BARNEY_NS::denoiser::optix::DENOISER_ALPHA_MODE_DEFAULT);
        
      OptixDeviceContext optixContext = owlContextGetOptixContext(device->owl,0);
      
      // Select denoiser model based on content type:
      // - LDR (0x2322): Optimized for Low Dynamic Range content (0-1 values)
      // - HDR (0x2323): Optimized for High Dynamic Range content (>1 values)
      // - AOV (0x2324): HDR with support for Arbitrary Output Variables
      // - TEMPORAL variants: Better for animation sequences with frame coherence
      OptixDenoiserModelKind modelKind = static_cast<OptixDenoiserModelKind>(
          BARNEY_NS::denoiser::optix::DENOISER_MODEL_KIND_DEFAULT);
          
      // Create the OptiX denoiser instance with the configured options
      // This validates the model kind and initializes internal denoiser state
      optixDenoiserCreate(optixContext, modelKind, &denoiserOptions, &denoiser);
    }
    
    Optix8Denoiser::~Optix8Denoiser()
    {
      cleanup();
      
      if (denoiser) {
        SetActiveGPU forDuration(device);
        optixDenoiserDestroy(denoiser);
        denoiser = {};
      }
    }
    
    void Optix8Denoiser::cleanup()
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
      if (out_rgba) {
        BARNEY_CUDA_CALL_NOTHROW(Free(out_rgba));
        out_rgba = 0;
      }
      if (in_normal) {
        BARNEY_CUDA_CALL_NOTHROW(Free(in_normal));
        in_normal = 0;
      }
      if (m_uintPixels) {
        BARNEY_CUDA_CALL_NOTHROW(Free(m_uintPixels));
        m_uintPixels = 0;
      }
    }
    
    void Optix8Denoiser::setup(vec2i size, void *pixelBuffer, int format)
    {
      init();
      SetActiveGPU forDuration(device);
      
      // Store pixel buffer and format for later use in launch()
      // This enhanced setup method allows external management of pixel buffers
      // and supports multiple pixel formats with automatic conversion
      m_pixelBuffer = pixelBuffer;
      m_format = format;
      
      // Query OptiX for memory requirements based on image dimensions
      // Memory needs scale roughly O(width * height) for the denoiser
      optixDenoiserComputeMemoryResources(denoiser, size.x, size.y, &denoiserSizes);
      
      // Calculate total memory requirements for allocation planning
      // This helps detect out-of-memory conditions before attempting allocation
      size_t totalMemoryNeeded = denoiserSizes.stateSizeInBytes + denoiserSizes.withoutOverlapScratchSizeInBytes;
      
      // Additional memory needed for non-FLOAT4 formats (pixel conversion buffer)
      // FLOAT4 can be processed directly, but RGBA8 formats need conversion space
      if (format != BARNEY_NS::denoiser::optix::pixel_format::FLOAT4) {
        totalMemoryNeeded += size_t(size.x) * size_t(size.y) * sizeof(uint32_t);
      }
      
      // Print memory requirements for debugging and capacity planning
      // This information helps users understand GPU memory usage and optimize settings
      bool showMemoryInfo = BARNEY_NS::denoiser::performance::MEMORY_REPORTING_DEFAULT || 
                           getenv("BARNEY_DENOISER_TIMING");
      if (showMemoryInfo) {
        printf("OptiX Denoiser memory requirements (%dx%d):\n", size.x, size.y);
        printf("  State: %zu MB (persistent denoiser state)\n", denoiserSizes.stateSizeInBytes / (1024*1024));
        printf("  Scratch: %zu MB (temporary computation space)\n", denoiserSizes.withoutOverlapScratchSizeInBytes / (1024*1024));
        if (format != BARNEY_NS::denoiser::optix::pixel_format::FLOAT4) {
          printf("  Pixel buffer: %zu MB (format conversion space)\n", (size_t(size.x) * size_t(size.y) * sizeof(uint32_t)) / (1024*1024));
        }
        printf("  Total: %zu MB\n", totalMemoryNeeded / (1024*1024));
      }
      
      // Allocate GPU memory buffers with comprehensive error handling
      // Memory allocation is done in dependency order to enable proper cleanup on failure
      
      // 1. Denoiser state memory: Persistent internal state used across invocations
      //    This stores learned parameters and intermediate data structures
      if (denoiserState) {
        BARNEY_CUDA_CALL(Free(denoiserState));
        denoiserState = 0;
      }
      
      try {
        BARNEY_CUDA_CALL(Malloc(&denoiserState, denoiserSizes.stateSizeInBytes));
      } catch (const std::exception& e) {
        std::cerr << "Failed to allocate denoiser state memory (" 
                  << denoiserSizes.stateSizeInBytes / (1024*1024) << " MB): " << e.what() << std::endl;
        throw;
      }
      
      // 2. Scratch memory: Temporary workspace for denoising computation
      //    This is used during optixDenoiserInvoke() and can be reused across frames
      if (denoiserScratch) {
        BARNEY_CUDA_CALL(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      
      try {
        BARNEY_CUDA_CALL(Malloc(&denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));
      } catch (const std::exception& e) {
        std::cerr << "Failed to allocate denoiser scratch memory (" 
                  << denoiserSizes.withoutOverlapScratchSizeInBytes / (1024*1024) << " MB): " << e.what() << std::endl;
        // Clean up previously allocated state memory to prevent memory leaks
        if (denoiserState) {
          BARNEY_CUDA_CALL_NOTHROW(Free(denoiserState));
          denoiserState = 0;
        }
        throw;
      }
      
      // 3. Pixel conversion buffer: Only needed for non-FLOAT4 formats
      //    This buffer stores converted RGBA8 output when the external buffer expects
      //    8-bit per channel format instead of 32-bit float per channel
      if (format != BARNEY_NS::denoiser::optix::pixel_format::FLOAT4) {
        if (m_uintPixels) {
          BARNEY_CUDA_CALL(Free(m_uintPixels));
          m_uintPixels = 0;
        }
        
        size_t pixelBufferSize = size_t(size.x) * size_t(size.y) * sizeof(uint32_t);
        try {
          BARNEY_CUDA_CALL(Malloc(&m_uintPixels, pixelBufferSize));
        } catch (const std::exception& e) {
          std::cerr << "Failed to allocate denoiser pixel buffer (" 
                    << pixelBufferSize / (1024*1024) << " MB): " << e.what() << std::endl;
          // Clean up all previously allocated memory to prevent leaks
          if (denoiserState) {
            BARNEY_CUDA_CALL_NOTHROW(Free(denoiserState));
            denoiserState = 0;
          }
          if (denoiserScratch) {
            BARNEY_CUDA_CALL_NOTHROW(Free(denoiserScratch));
            denoiserScratch = 0;
          }
          throw;
        }
      }
      
      // Initialize the denoiser with allocated memory buffers
      // This prepares internal data structures and validates memory layout
      optixDenoiserSetup(denoiser,
                         0, // stream (use default CUDA stream)
                         size.x,
                         size.y,
                         (CUdeviceptr)denoiserState,
                         denoiserSizes.stateSizeInBytes,
                         (CUdeviceptr)denoiserScratch,
                         denoiserSizes.withoutOverlapScratchSizeInBytes);
      
      // Configure input image layer for denoising operation
      // OptiX denoiser works with FLOAT4 format internally, regardless of external format
      layer.input.data = (CUdeviceptr)pixelBuffer;           // Source image data pointer
      layer.input.width = size.x;                            // Image width in pixels
      layer.input.height = size.y;                           // Image height in pixels
      layer.input.pixelStrideInBytes = 0;                    // Dense packing (no gaps between pixels)
      layer.input.rowStrideInBytes = 4 * sizeof(float) * size.x; // Bytes per row (4 floats * width)
      layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;        // 32-bit float per channel format
      
      // Configure output image layer (same as input for in-place denoising)
      // The denoiser writes the cleaned image back to the same buffer location
      memcpy(&layer.output, &layer.input, sizeof(layer.output));
      layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY; // Standard beauty render AOV type
      
      // Store image dimensions for later use in launch() method
      numPixels = size;
    }
    
    void Optix8Denoiser::resize(vec2i dims)
    {
      // Enhanced resize that preserves format and pixel buffer if already set up
      if (m_pixelBuffer) {
        setup(dims, m_pixelBuffer, m_format);
        return;
      }
      
      // Original behavior for backward compatibility
      this->numPixels = dims;
      SetActiveGPU forDuration(device);

      denoiserSizes.overlapWindowSizeInPixels = 0;
      optixDenoiserComputeMemoryResources(denoiser, dims.x, dims.y, &denoiserSizes);
      
      // Free existing buffers
      if (denoiserScratch) {
        BARNEY_CUDA_CALL(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      if (denoiserState) {
        BARNEY_CUDA_CALL(Free(denoiserState));
        denoiserState = 0;
      }
      if (in_rgba) {
        BARNEY_CUDA_CALL(Free(in_rgba));
        in_rgba = 0;
      }
      if (out_rgba) {
        BARNEY_CUDA_CALL(Free(out_rgba));
        out_rgba = 0;
      }
      if (in_normal) {
        BARNEY_CUDA_CALL(Free(in_normal));
        in_normal = 0;
      }
      
      // Allocate new buffers for backward compatibility
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));
      BARNEY_CUDA_CALL(Malloc(&denoiserState, denoiserSizes.stateSizeInBytes));
      BARNEY_CUDA_CALL(Malloc(&in_rgba, dims.x*dims.y*sizeof(*in_rgba)));
      BARNEY_CUDA_CALL(Malloc(&out_rgba, dims.x*dims.y*sizeof(*out_rgba)));
      BARNEY_CUDA_CALL(Malloc(&in_normal, dims.x*dims.y*sizeof(*in_normal)));
      
      optixDenoiserSetup(denoiser,
                         0, // stream
                         dims.x,
                         dims.y,
                         (CUdeviceptr)denoiserState,
                         denoiserSizes.stateSizeInBytes,
                         (CUdeviceptr)denoiserScratch,
                         denoiserSizes.withoutOverlapScratchSizeInBytes);
    }
    
    void Optix8Denoiser::launch()
    {
      Timer timer("optixDenoiserInvoke()");
      SetActiveGPU forDuration(device);
      
      // Configure temporal blending factor for multi-frame denoising
      // A value of 0.0 uses only the current frame (no temporal blending)
      // Higher values blend more with previous frames for smoother animation
      if (params.blendFactor == 0.0f) {
        params.blendFactor = BARNEY_NS::denoiser::optix::ENHANCED_BLEND_FACTOR_DEFAULT;
      }
      
      // Execute the OptiX denoising algorithm on GPU
      // This is the core denoising computation that processes the image
      optixDenoiserInvoke(denoiser,
                          0, // stream (use default CUDA stream for simplicity)
                          &params,               // Runtime parameters (blend factor, etc.)
                          (CUdeviceptr)denoiserState,    // Persistent denoiser state
                          static_cast<unsigned int>(denoiserSizes.stateSizeInBytes),
                          &guideLayer,           // Guide images (albedo, normal, etc.)
                          &layer,                // Input/output image layers
                          1,                     // Number of layers to process
                          0, // input offset X     // Tile offset (0 for full-image denoising)
                          0, // input offset Y     // Tile offset (0 for full-image denoising)
                          (CUdeviceptr)denoiserScratch,  // Temporary computation workspace
                          static_cast<unsigned int>(denoiserSizes.withoutOverlapScratchSizeInBytes));
      
      // Post-process: Convert denoised FLOAT4 output to requested format
      // OptiX denoiser always outputs FLOAT4, but external applications may need
      // different formats like RGBA8 (uint32) or sRGB-encoded RGBA8
      if (m_format != BARNEY_NS::denoiser::optix::pixel_format::FLOAT4 && m_uintPixels) {
        Timer pixel_timer("denoiser transform pixels");
        auto numPixels_total = size_t(layer.output.width) * size_t(layer.output.height);
        
#ifdef BARNEY_HAVE_THRUST
        // Use Thrust library for parallel GPU-based pixel format conversion
        // Thrust provides optimized parallel algorithms that run efficiently on GPU
        auto begin = thrust::device_ptr<vec4f>((vec4f *)m_pixelBuffer);
        auto end = begin + numPixels_total;
        
        // Convert float4 to RGBA8 with optional sRGB gamma correction
        if (m_format == BARNEY_NS::denoiser::optix::pixel_format::UFIXED8_RGBA_SRGB) {
          thrust::transform(thrust::cuda::par,
                            begin,
                            end,
                            thrust::device_pointer_cast<uint32_t>((uint32_t*)m_uintPixels),
                            [] __device__(const vec4f &in) {
                              // Apply sRGB gamma correction for display-ready output
                              // sRGB is the standard color space for most displays and web content
                              vec4f srgb;
                              srgb.x = (in.x <= 0.0031308f) ? 12.92f * in.x : 1.055f * powf(in.x, 1.f/2.4f) - 0.055f;
                              srgb.y = (in.y <= 0.0031308f) ? 12.92f * in.y : 1.055f * powf(in.y, 1.f/2.4f) - 0.055f;
                              srgb.z = (in.z <= 0.0031308f) ? 12.92f * in.z : 1.055f * powf(in.z, 1.f/2.4f) - 0.055f;
                              srgb.w = in.w; // Alpha remains linear
                              
                              // Pack 4 floats into single 32-bit RGBA8 value (8 bits per channel)
                              uint32_t r = min(255, max(0, int(srgb.x * 256.f)));
                              uint32_t g = min(255, max(0, int(srgb.y * 256.f)));
                              uint32_t b = min(255, max(0, int(srgb.z * 256.f)));
                              uint32_t a = min(255, max(0, int(srgb.w * 256.f)));
                              return (r << 0) | (g << 8) | (b << 16) | (a << 24); // RGBA byte order
                            });
        } else { // UFIXED8_RGBA format (linear, no gamma correction)
          thrust::transform(thrust::cuda::par,
                            begin,
                            end,
                            thrust::device_pointer_cast<uint32_t>((uint32_t*)m_uintPixels),
                            [] __device__(const vec4f &in) {
                              // Direct linear to 8-bit conversion (no gamma correction)
                              // Used when the display pipeline will handle gamma correction separately
                              uint32_t r = min(255, max(0, int(in.x * 256.f)));
                              uint32_t g = min(255, max(0, int(in.y * 256.f)));
                              uint32_t b = min(255, max(0, int(in.z * 256.f)));
                              uint32_t a = min(255, max(0, int(in.w * 256.f)));
                              return (r << 0) | (g << 8) | (b << 16) | (a << 24); // RGBA byte order
                            });
        }
#else
        // Fallback: use custom CUDA kernel for pixel conversion when Thrust is unavailable
        // This provides the same functionality as the Thrust version but with a custom kernel
        convert_float4_to_rgba(m_pixelBuffer,           // Input: denoised float4 pixels
                               m_uintPixels,            // Output: converted uint32 pixels  
                               layer.output.width,     // Image width
                               layer.output.height,    // Image height
                               m_format == BARNEY_NS::denoiser::optix::pixel_format::UFIXED8_RGBA_SRGB, // Apply sRGB?
                               0); // Use default CUDA stream
#endif
      }
    }
    
    void Optix8Denoiser::run(float blendFactor)
    {
      // Enhanced run method that uses the new launch() method when configured with setup()
      if (m_pixelBuffer) {
        params.blendFactor = blendFactor;
        launch();
      } else {
        // Original behavior for backward compatibility - this is what FrameBuffer currently uses
        Timer timer("optixDenoiserInvoke()");
        SetActiveGPU forDuration(device);
        OptixDenoiserLayer local_layer = {};
        
        local_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        local_layer.input.rowStrideInBytes = numPixels.x*sizeof(vec4f);
        local_layer.input.pixelStrideInBytes = sizeof(vec4f);
        local_layer.input.width  = numPixels.x;
        local_layer.input.height = numPixels.y;
        local_layer.input.data   = (CUdeviceptr)in_rgba;
        
        OptixDenoiserGuideLayer local_guideLayer = {};
        local_guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        local_guideLayer.normal.rowStrideInBytes = numPixels.x*sizeof(vec3f);
        local_guideLayer.normal.pixelStrideInBytes = sizeof(vec3f);
        local_guideLayer.normal.width  = numPixels.x;
        local_guideLayer.normal.height = numPixels.y;
        local_guideLayer.normal.data = (CUdeviceptr)in_normal;
        
        local_layer.output = local_layer.input;
        local_layer.output.data = (CUdeviceptr)out_rgba;

        OptixDenoiserParams local_params = {};
        local_params.blendFactor = blendFactor;

        optixDenoiserInvoke(denoiser,
                            0,
                            &local_params,
                            (CUdeviceptr)denoiserState,
                            denoiserSizes.stateSizeInBytes,
                            &local_guideLayer,
                            &local_layer,
                            1,
                            0,
                            0,
                            (CUdeviceptr)denoiserScratch,
                            denoiserSizes.withoutOverlapScratchSizeInBytes);
      }
    }
    
    void *Optix8Denoiser::mapColorBuffer()
    {
      if (m_format == BARNEY_NS::denoiser::optix::pixel_format::FLOAT4) {
        // Copy from device to host (would need host buffer)
        return m_pixelBuffer;
      } else {
        // Return the converted uint pixel buffer
        return m_uintPixels;
      }
    }
    
    void *Optix8Denoiser::mapGPUColorBuffer()
    {
      return m_format == BARNEY_NS::denoiser::optix::pixel_format::FLOAT4 ? m_pixelBuffer : m_uintPixels;
    }
    
#endif
    
  }
}

