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

#pragma once

#include "rtcore/optix/Device.h"

namespace rtc {
  namespace optix {

    /*! abstract interface to a denoiser. implementation(s) depend of
        which optix version and/or oidn are available */
    struct Denoiser {
      Denoiser(Device* device) : device(device) {}
      virtual ~Denoiser() = default;
      virtual void resize(vec2i dims) = 0;
      virtual void run(float blendFactor) = 0;
      vec4f *out_rgba  = 0;
      vec4f *in_rgba   = 0;
      vec3f *in_normal = 0;
      Device* const device;
    };

#if OPTIX_VERSION >= 80000
    /*! Enhanced OptiX 8 denoiser implementation with RTX features
        
        Based on VisRTX implementation with improved memory management,
        better pixel format handling, and sRGB support.
        
        Key features:
        - Multiple pixel format support (FLOAT4, RGBA8, RGBA8_SRGB)
        - Advanced memory management with proper error handling
        - Temporal blending for animation sequences
        - Performance instrumentation and memory reporting
        - Backward compatibility with existing Barney code
        
        Usage patterns:
        1. Legacy mode: FrameBuffer calls resize() + run() - uses internal buffers
        2. Enhanced mode: External code calls setup() + launch() - uses external buffers
        
        Memory buffers:
        - State: Persistent denoiser parameters across frames (~50MB for 1080p)
        - Scratch: Temporary workspace during computation (~150MB for 1080p)  
        - Pixel: Format conversion for non-FLOAT4 outputs (~8MB for 1080p)
    */
    struct Optix8Denoiser : public Denoiser {
      Optix8Denoiser(Device *device);
      virtual ~Optix8Denoiser();
      
      // Legacy interface (used by FrameBuffer for backward compatibility)
      void resize(vec2i dims) override;
      void run(float blendFactor) override;
      
      // Enhanced interface (inspired by VisRTX RTX denoiser)
      void setup(vec2i size, void *pixelBuffer, int format);  ///< Configure with external buffer  
      void cleanup();                                         ///< Clean up all allocated memory
      void launch();                                          ///< Execute denoising with format conversion
      void *mapColorBuffer();                                 ///< Get host-accessible output buffer
      void *mapGPUColorBuffer();                              ///< Get GPU-accessible output buffer
      
      // Core OptiX denoiser objects and configuration
      vec2i                numPixels;          ///< Current image dimensions
      OptixDenoiser        denoiser = {};      ///< OptiX denoiser instance handle
      OptixDenoiserOptions denoiserOptions;   ///< Configuration (guide layers, alpha mode)
      OptixDenoiserParams  params = {};       ///< Runtime parameters (blend factor)
      OptixDenoiserGuideLayer guideLayer = {}; ///< Guide images (albedo, normal, flow)
      OptixDenoiserLayer   layer = {};        ///< Input/output image layer configuration
      
      // GPU memory buffers (allocated by OptiX based on image size)
      void                *denoiserScratch = 0; ///< Temporary computation workspace
      void                *denoiserState   = 0; ///< Persistent denoiser state across frames
      OptixDenoiserSizes   denoiserSizes;     ///< Memory size requirements from OptiX
      
      // Enhanced memory management for multiple pixel formats
      void                *m_pixelBuffer = nullptr; ///< External pixel buffer (FLOAT4 format)
      int                  m_format = 0;           ///< Pixel format: 0=unknown, 1=float4, 2=uint32, 3=uint32_srgb
      void                *m_uintPixels = nullptr;  ///< Converted output buffer for RGBA8 formats
      
    private:
      void init();
    };
#endif
    
  }
}

