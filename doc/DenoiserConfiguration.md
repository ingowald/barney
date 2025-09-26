# Barney Denoiser Configuration Guide

This document explains how to configure denoiser parameters in Barney using the centralized configuration system.

## Configuration Location

All denoiser default values are centralized in:
```
barney/common/DenoiserConfig.h
```

**This is the single file to modify for changing any denoiser defaults.**

## Configuration Categories

### 1. ANARI Parameters

```cpp
namespace BARNEY_NS::denoiser {
    // ANARI renderer "denoise" parameter default
    constexpr bool ANARI_DENOISE_DEFAULT = true;
}
```

**Usage**: Controls the default value for `anari::setParameter(device, renderer, "denoise", value)`

### 2. FrameBuffer Parameters

```cpp
namespace BARNEY_NS::denoiser {
    // FrameBuffer enableDenoising default
    constexpr bool FRAMEBUFFER_ENABLE_DENOISING_DEFAULT = true;
    
    // Temporal blending calculation
    constexpr float BLEND_FACTOR_OFFSET = 100.0f;
    constexpr int BLEND_FACTOR_MIN_ACCUM_ID = 1;
}
```

**Blend Factor Formula**: `blendFactor = (accumID - 1) / (accumID + BLEND_FACTOR_OFFSET)`

**Effect of BLEND_FACTOR_OFFSET**:
- **Lower values** (e.g., 50.0f): Faster temporal convergence, less stability
- **Higher values** (e.g., 200.0f): Slower convergence, more temporal stability
- **Default 100.0f**: Balanced approach

### 3. OptiX Backend Parameters

```cpp
namespace BARNEY_NS::denoiser::optix {
    // Guide layer configuration
    constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 0;  // 0=disabled, 1=enabled
    constexpr unsigned int GUIDE_NORMAL_DEFAULT = 1;  // 0=disabled, 1=enabled
    
    // Model type: 0=LDR, 1=HDR
    constexpr int DENOISER_MODEL_KIND_DEFAULT = 0;    // LDR
    
    // Alpha handling: 0=COPY, 1=DENOISE  
    constexpr int DENOISER_ALPHA_MODE_DEFAULT = 1;    // DENOISE
    
    // Enhanced mode blend factor
    constexpr float ENHANCED_BLEND_FACTOR_DEFAULT = 0.0f;
    
    // Pixel format constants
    namespace pixel_format {
        constexpr int UNKNOWN = 0;
        constexpr int FLOAT4 = 1;
        constexpr int UFIXED8_RGBA = 2; 
        constexpr int UFIXED8_RGBA_SRGB = 3;
    }
}
```

### 4. OIDN (CPU) Backend Parameters

```cpp
namespace BARNEY_NS::denoiser::oidn {
    // HDR processing mode
    constexpr bool HDR_MODE_DEFAULT = true;
    
    // Filter type
    constexpr const char* FILTER_TYPE_DEFAULT = "RT";  // or "RTLightmap"
}
```

### 5. Performance Parameters

```cpp
namespace BARNEY_NS::denoiser::performance {
    // Timing instrumentation
    constexpr bool TIMING_ENABLED_DEFAULT = false;
    
    // Verbosity level
    constexpr int VERBOSITY_LEVEL_DEFAULT = 0;  // 0=silent, 1=basic, 2=detailed
}
```

## Common Configuration Changes

### Enable HDR Denoising by Default

```cpp
// In DenoiserConfig.h
constexpr int DENOISER_MODEL_KIND_DEFAULT = 0x2323; // Change from 0x2322 (LDR) to 0x2323 (HDR)
```

### Enable Albedo Guide Layer

```cpp
// In DenoiserConfig.h  
constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 1; // Change from 0 to 1
```

### Faster Temporal Convergence

```cpp
// In DenoiserConfig.h
constexpr float BLEND_FACTOR_OFFSET = 50.0f; // Change from 100.0f
```

### Disable Denoising by Default

```cpp
// In DenoiserConfig.h
constexpr bool ANARI_DENOISE_DEFAULT = false; // Change from true
constexpr bool FRAMEBUFFER_ENABLE_DENOISING_DEFAULT = false; // Change from true
```

### Enable Performance Timing by Default

```cpp
// In DenoiserConfig.h
constexpr bool TIMING_ENABLED_DEFAULT = true; // Change from false
```

## Utility Functions

The configuration header also provides validation and calculation utilities:

```cpp
// Validate blend factor range
bool isValidBlendFactor(float blendFactor);

// Calculate blend factor from accumulation ID
float calculateBlendFactor(int accumID);

// Validate pixel format
bool isValidPixelFormat(int format);
```

## Environment Variable Overrides

Some settings can still be overridden at runtime:

```bash
# Enable performance timing (overrides TIMING_ENABLED_DEFAULT)
export BARNEY_DENOISER_TIMING=1

# Disable denoising entirely
export BARNEY_CONFIG="denoising=0"

# Force CPU backend (uses OIDN instead of OptiX)
export BARNEY_FORCE_CPU=1
```

## Build-Time Configuration

CMake options override configuration file settings:

```bash
# Disable all denoising support
cmake -DBARNEY_DISABLE_DENOISING=ON

# Disable OptiX denoising specifically  
cmake -DOPTIX_DISABLE_DENOISING=ON
```

## Backward Compatibility

The centralized configuration maintains full backward compatibility:

- **Existing applications**: No changes required
- **Runtime parameters**: Still work as before
- **Environment variables**: Still override defaults
- **Performance**: No performance impact

## Best Practices

1. **Change defaults in `DenoiserConfig.h`** rather than hardcoding values
2. **Test configuration changes** with different scenes and backends
3. **Document any changes** to defaults for team members
4. **Use validation functions** when implementing custom parameter handling
5. **Consider performance impact** of enabling additional guide layers

## Configuration Scenarios

### High-Quality Mode (Slower)
```cpp
constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 1;    // Enable albedo guide
constexpr unsigned int GUIDE_NORMAL_DEFAULT = 1;    // Enable normal guide  
constexpr int DENOISER_MODEL_KIND_DEFAULT = 0x2323; // Use HDR model (OptiX constant)
constexpr float BLEND_FACTOR_OFFSET = 200.0f;       // More temporal stability
```

### Performance Mode (Faster)
```cpp
constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 0;    // Disable albedo guide
constexpr unsigned int GUIDE_NORMAL_DEFAULT = 0;    // Disable normal guide
constexpr int DENOISER_MODEL_KIND_DEFAULT = 0x2322; // Use LDR model (OptiX constant)
constexpr float BLEND_FACTOR_OFFSET = 50.0f;        // Faster convergence
```

### Debug Mode
```cpp
constexpr bool TIMING_ENABLED_DEFAULT = true;       // Enable timing
constexpr int VERBOSITY_LEVEL_DEFAULT = 2;          // Detailed logging
constexpr bool MEMORY_REPORTING_DEFAULT = true;     // Show memory usage
```

### Low Memory Mode (For GPUs with Limited VRAM)
```cpp
constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 0;            // Disable albedo guide
constexpr unsigned int GUIDE_NORMAL_DEFAULT = 0;            // Disable normal guide  
constexpr bool DISABLE_ON_ALLOCATION_FAILURE = true;        // Continue without denoising on OOM
constexpr bool FALLBACK_TO_CPU_ON_GPU_OOM = true;          // Try CPU denoiser if GPU fails
```

## Troubleshooting

### "Unknown model kind: 0x0" Error

This error occurs when the denoiser model kind is not set to a valid OptiX constant. Make sure to use:

```cpp
// Correct - use actual OptiX constants
constexpr int DENOISER_MODEL_KIND_DEFAULT = 0x2322; // LDR
constexpr int DENOISER_MODEL_KIND_DEFAULT = 0x2323; // HDR

// Wrong - simple integers don't work
constexpr int DENOISER_MODEL_KIND_DEFAULT = 0; // Will cause "Unknown model kind" error
```

### "Out of Memory" Errors

OptiX denoisers require significant GPU memory. Memory usage scales with image resolution:

- **1920x1080**: ~200-400 MB
- **3840x2160 (4K)**: ~800-1600 MB  
- **7680x4320 (8K)**: ~3-6 GB

**Solutions:**

1. **Reduce image resolution** during development
2. **Disable guide layers** to reduce memory usage:
   ```cpp
   constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 0;
   constexpr unsigned int GUIDE_NORMAL_DEFAULT = 0;
   ```
3. **Enable memory reporting** to see actual requirements:
   ```bash
   export BARNEY_DENOISER_TIMING=1
   ```
4. **Configure fallback behavior**:
   ```cpp
   constexpr bool FALLBACK_TO_CPU_ON_GPU_OOM = true;
   constexpr bool DISABLE_ON_ALLOCATION_FAILURE = true;
   ```

This centralized approach makes it easy to experiment with different denoising configurations and maintain consistent defaults across the entire Barney system.
