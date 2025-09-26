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

/*! \file DenoiserConfig.h
    
    Centralized configuration for all denoiser parameters in Barney.
    
    This header provides a single source of truth for all denoising-related
    default values, making it easy to adjust behavior across the entire system.
    
    Key benefits:
    - Single point of configuration for easy maintenance
    - Compile-time constants for zero runtime overhead  
    - Comprehensive documentation of parameter effects
    - Type safety and validation utilities
    
    Usage:
    - Modify values in this file to change system-wide defaults
    - Use the validation utilities for custom parameter handling
    - Refer to doc/DenoiserConfiguration.md for detailed usage guide
    
    Categories:
    - ANARI: High-level API parameters
    - FrameBuffer: Runtime rendering parameters
    - OptiX: GPU denoiser backend configuration
    - OIDN: CPU denoiser backend configuration  
    - Performance: Timing and reporting options
    - Memory: Allocation and fallback behavior
*/

#pragma once

namespace BARNEY_NS {
  namespace denoiser {

    // ==================================================================
    // ANARI Default Parameters
    // ==================================================================
    
    /*! Default value for ANARI renderer "denoise" parameter */
    constexpr bool ANARI_DENOISE_DEFAULT = true;

    // ==================================================================
    // FrameBuffer Default Parameters  
    // ==================================================================

    /*! Default enableDenoising value when not explicitly set */
    constexpr bool FRAMEBUFFER_ENABLE_DENOISING_DEFAULT = true;

    /*! Blend factor calculation parameters for temporal denoising
        Formula: blendFactor = (accumID - 1) / (accumID + BLEND_FACTOR_OFFSET)
        
        Lower BLEND_FACTOR_OFFSET = faster convergence, less temporal stability
        Higher BLEND_FACTOR_OFFSET = slower convergence, more temporal stability */
    constexpr float BLEND_FACTOR_OFFSET = 20.0f;

    /*! Minimum accumulation ID before temporal blending starts */
    constexpr int BLEND_FACTOR_MIN_ACCUM_ID = 0;

    // ==================================================================
    // OptiX Denoiser Default Parameters
    // ==================================================================

    namespace optix {
      /*! Guide albedo layer usage (0 = disabled, 0 = enabled)
          Albedo guide can improve denoising quality but requires additional memory */
      constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 1;

      /*! Guide normal layer usage (0 = disabled, 1 = enabled)  
          Normal guide typically improves denoising quality significantly */
      constexpr unsigned int GUIDE_NORMAL_DEFAULT = 1;

      /*! Default denoiser model type
          Use actual OptiX constants, not simple integers:
          OPTIX_DENOISER_MODEL_KIND_LDR (0x2322) = Low Dynamic Range (better compatibility)
          OPTIX_DENOISER_MODEL_KIND_HDR (0x2323) = High Dynamic Range (better quality for HDR content)
          OPTIX_DENOISER_MODEL_KIND_AOV (0x2324) = HDR with AOVs support
          OPTIX_DENOISER_MODEL_KIND_TEMPORAL (0x2325) = HDR temporally stable
          OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV (0x2326) = HDR AOVs temporally stable */
      constexpr int DENOISER_MODEL_KIND_DEFAULT = 0x2322; // LDR

      /*! Default alpha channel handling  
          Use actual OptiX constants:
          OPTIX_DENOISER_ALPHA_MODE_COPY (0) = preserve alpha unchanged
          OPTIX_DENOISER_ALPHA_MODE_DENOISE (1) = apply denoising to alpha channel too */
      constexpr int DENOISER_ALPHA_MODE_DEFAULT = 1; // DENOISE

      /*! Default blend factor when using enhanced setup() method */
      constexpr float ENHANCED_BLEND_FACTOR_DEFAULT = 0.0f;

      /*! Pixel format constants for enhanced denoiser */
      namespace pixel_format {
        constexpr int UNKNOWN = 0;
        constexpr int FLOAT4 = 1;
        constexpr int UFIXED8_RGBA = 2; 
        constexpr int UFIXED8_RGBA_SRGB = 3;
      }
    }

    // ==================================================================
    // OIDN (CPU) Denoiser Default Parameters
    // ==================================================================

    namespace oidn {
      /*! Default HDR processing mode for OIDN
          true = process as HDR images (better for ray-traced content)
          false = process as LDR images */
      constexpr bool HDR_MODE_DEFAULT = true;

      /*! Default OIDN filter type
          "RT" = Ray-traced filter (optimized for path-traced images)
          "RTLightmap" = Lightmap filter (for baked lighting) */
      constexpr const char* FILTER_TYPE_DEFAULT = "RT";
    }

    // ==================================================================
    // Performance and Debugging Defaults
    // ==================================================================

    namespace performance {
      /*! Default timing instrumentation state 
          Can be overridden by BARNEY_DENOISER_TIMING environment variable */
      constexpr bool TIMING_ENABLED_DEFAULT = false;

      /*! Default verbosity level for denoiser operations */
      constexpr int VERBOSITY_LEVEL_DEFAULT = 0; // 0=silent, 1=basic, 2=detailed

      /*! Enable memory usage reporting by default 
          Shows memory requirements during denoiser setup */
      constexpr bool MEMORY_REPORTING_DEFAULT = true;
    }

    // ==================================================================
    // Memory Management Defaults
    // ==================================================================

    namespace memory {
      /*! Fallback to CPU denoiser if GPU memory allocation fails
          When true, will try OIDN CPU denoiser if OptiX GPU allocation fails */
      constexpr bool FALLBACK_TO_CPU_ON_GPU_OOM = false;

      /*! Disable denoising entirely if memory allocation fails
          When true, continues without denoising instead of failing */
      constexpr bool DISABLE_ON_ALLOCATION_FAILURE = true;
    }

    // ==================================================================
    // Environment Variable Names (for reference)
    // ==================================================================

    namespace env_vars {
      constexpr const char* DENOISER_TIMING = "BARNEY_DENOISER_TIMING";
      constexpr const char* FORCE_CPU = "BARNEY_FORCE_CPU";
      constexpr const char* CONFIG_DENOISING = "denoising"; // Used in BARNEY_CONFIG
      constexpr const char* CONFIG_SKIP_DENOISING = "SKIP_DENOISING"; // Used in BARNEY_CONFIG
    }

    // ==================================================================
    // Utility Functions for Parameter Validation
    // ==================================================================

    /*! Validate blend factor is in valid range [0.0, 1.0] */
    inline bool isValidBlendFactor(float blendFactor) {
      return blendFactor >= 0.0f && blendFactor <= 1.0f;
    }

    /*! Calculate default blend factor based on accumulation ID */
    inline float calculateBlendFactor(int accumID) {
      if (accumID < BLEND_FACTOR_MIN_ACCUM_ID) return 0.0f;
      return static_cast<float>(accumID - 1) / static_cast<float>(accumID + BLEND_FACTOR_OFFSET);
    }

    /*! Validate pixel format is supported */
    inline bool isValidPixelFormat(int format) {
      return format >= optix::pixel_format::UNKNOWN && format <= optix::pixel_format::UFIXED8_RGBA_SRGB;
    }

  } // namespace denoiser
} // namespace BARNEY_NS
