# RTX Denoiser Integration in Barney

This document describes the integration of the VisRTX denoiser functionality into Barney's OptiX backend.

## Overview

The RTX denoiser integration enhances Barney's existing OptiX denoiser with advanced features from the VisRTX implementation, including:

- **Improved memory management** with RAII-based device buffers
- **Enhanced pixel format handling** with automatic conversion support
- **Advanced sRGB conversion** using either Thrust or custom CUDA kernels
- **Performance instrumentation** for monitoring denoiser performance
- **Better error handling** and resource cleanup

## Key Enhancements

### 1. Enhanced Denoiser Interface

The `Optix8Denoiser` class now includes additional methods inspired by the RTX implementation:

```cpp
// Enhanced setup method with format support
void setup(vec2i size, void *pixelBuffer, int format);

// Better resource cleanup
void cleanup();

// Direct launch method with instrumentation
void launch();

// Buffer mapping methods
void *mapColorBuffer();
void *mapGPUColorBuffer();
```

### 2. Improved Memory Management

- **DeviceBuffer class**: Simple RAII-based device memory management
- **Automatic cleanup**: Resources are properly freed in destructor
- **Format-aware allocation**: Different pixel formats are handled automatically

### 3. Advanced Pixel Format Support

The enhanced denoiser supports multiple pixel formats:

- **Format 1**: `FLOAT4` - Native float4 format (no conversion needed)
- **Format 2**: `UFIXED8_RGBA` - 8-bit RGBA with linear color space
- **Format 3**: `UFIXED8_RGBA_SRGB` - 8-bit RGBA with sRGB conversion

### 4. Dual-Path Pixel Conversion

**Thrust Path (when available):**
```cpp
#ifdef BARNEY_HAVE_THRUST
thrust::transform(thrust::cuda::par,
                  begin, end, output,
                  [](const vec4f &in) { /* conversion logic */ });
#endif
```

**CUDA Kernel Fallback:**
```cpp
convert_float4_to_rgba(pixelBuffer, output, width, height, srgb, stream);
```

### 5. Performance Instrumentation

Timing information can be enabled with the environment variable:
```bash
export BARNEY_DENOISER_TIMING=1
```

This will output timing information for:
- `optixDenoiserInvoke()` - Main denoising operation
- `denoiser transform pixels` - Pixel format conversion

## Implementation Details

### File Structure

```
rtcore/optix/
├── Denoiser.h          # Enhanced denoiser interface
├── Denoiser.cpp        # Main implementation with RTX features
├── DenoiserUtils.h     # Utility function declarations
├── DenoiserUtils.cu    # CUDA kernels for pixel conversion
└── DeviceBuffer.h      # RAII device memory management
```

### RTX Features Integrated

1. **Memory Management**:
   - Based on VisRTX `DeviceBuffer` class
   - Automatic cleanup and error handling
   - Efficient memory allocation patterns

2. **Pixel Format Handling**:
   - Support for multiple output formats
   - Automatic conversion between linear and sRGB color spaces
   - Efficient GPU-based conversion using Thrust or custom kernels

3. **Performance Monitoring**:
   - Timing instrumentation similar to VisRTX
   - Optional performance logging
   - Memory usage tracking

4. **Error Handling**:
   - Proper resource cleanup on errors
   - Exception safety with RAII
   - Graceful fallbacks for unsupported features

## Usage

### ANARI Integration (Recommended)

The RTX denoiser integrates seamlessly with ANARI. To enable denoising:

```cpp
// C++ ANARI API
anari::setParameter(device, renderer, "denoise", true);
anari::commitParameters(device, renderer);

// C ANARI API  
anariSetParameter(device, renderer, "denoise", ANARI_BOOL, &enableDenoising);
anariCommitParameters(device, renderer);
```

**This is the standard way to activate denoising and works automatically with the RTX enhancements.**

### Internal Flow

When `anari::setParameter(device, renderer, "denoise", true)` is called:

1. **Renderer**: `Renderer::commitParameters()` sets `m_denoise = true`
2. **Frame**: `Frame::finalize()` gets `denoise |= m_renderer->denoise()`
3. **FrameBuffer**: `bnSet1i(m_bnFrameBuffer, "enableDenoising", denoise)` 
4. **Denoising**: `FrameBuffer::readColorChannel()` checks `enableDenoising` and calls `denoiser->run()`

### Basic Usage (Low-level API)

For direct access to the enhanced denoiser:

```cpp
auto denoiser = device->createDenoiser();
denoiser->resize(vec2i(width, height));
denoiser->run(blendFactor);
```

### Enhanced Usage

For advanced features, use the new setup method:

```cpp
auto denoiser = device->createDenoiser();

// Setup with specific format and pixel buffer
denoiser->setup(vec2i(width, height), pixelBuffer, format);

// Configure denoising parameters
denoiser->params.blendFactor = 0.5f;

// Run denoising with instrumentation
denoiser->launch();

// Access results
void* result = denoiser->mapGPUColorBuffer();
```

### Format Codes

- `1` - FLOAT4: Native float4 format
- `2` - UFIXED8_RGBA: 8-bit RGBA linear
- `3` - UFIXED8_RGBA_SRGB: 8-bit RGBA with sRGB conversion

## Performance Considerations

1. **Thrust vs Custom Kernels**:
   - Thrust provides more optimized conversions when available
   - Custom kernels provide fallback compatibility
   - Both implementations are GPU-accelerated

2. **Memory Allocation**:
   - DeviceBuffer uses RAII for automatic cleanup
   - Memory is allocated once and reused when possible
   - Separate buffers for different pixel formats

3. **Format Conversion Overhead**:
   - FLOAT4 format has no conversion overhead
   - 8-bit formats require GPU-based conversion
   - sRGB conversion adds minimal computational cost

## Configuration Options

### Centralized Configuration

**All denoiser default values are now centralized in:**
```
barney/common/DenoiserConfig.h
```

This single file controls all denoiser defaults. See `doc/DenoiserConfiguration.md` for detailed configuration options.

### Build-time Configuration

The RTX denoiser features are automatically enabled when:
- OptiX backend is enabled (`BARNEY_BACKEND_OPTIX=ON`)
- OptiX version >= 8.0
- Denoising is not disabled (`BARNEY_DISABLE_DENOISING=OFF`)

### Runtime Configuration

Environment variables:
- `BARNEY_DENOISER_TIMING=1` - Enable performance timing output
- `BARNEY_FORCE_CPU` - Force CPU backend (disables OptiX denoiser)
- `BARNEY_CONFIG="denoising=0"` - Disable denoising entirely

### Quick Configuration Changes

Common configuration examples in `DenoiserConfig.h`:

```cpp
// Enable HDR denoising by default
constexpr int DENOISER_MODEL_KIND_DEFAULT = 1; // 0=LDR, 1=HDR

// Enable albedo guide layer  
constexpr unsigned int GUIDE_ALBEDO_DEFAULT = 1; // 0=disabled, 1=enabled

// Faster temporal convergence
constexpr float BLEND_FACTOR_OFFSET = 50.0f; // Lower = faster convergence
```

## Compatibility

The RTX denoiser integration maintains full backward compatibility:

- **Existing ANARI code continues to work without modifications**
- `anari::setParameter(device, renderer, "denoise", true)` works exactly as before
- Original API methods are preserved and enhanced
- Performance characteristics are maintained or improved
- Memory usage patterns are optimized
- Enhanced features are automatically available when using the OptiX backend

### Verified Compatibility

✅ **ANARI Parameter Setting**: `anari::setParameter(device, renderer, "denoise", true)`  
✅ **Backward Compatibility**: All existing denoiser API calls work unchanged  
✅ **Performance**: Enhanced or equivalent performance in all cases  
✅ **Memory Safety**: Improved RAII-based memory management  
✅ **Error Handling**: Better error recovery and resource cleanup

## Future Enhancements

Potential future improvements:
1. **Albedo and Normal Guide Support**: Add support for additional guide layers
2. **Temporal Denoising**: Implement frame-to-frame temporal filtering
3. **Adaptive Quality**: Dynamic quality adjustment based on performance
4. **Multi-GPU Support**: Enhanced denoising across multiple GPUs

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Ensure OptiX SDK is properly installed and CUDA is available
2. **Runtime Errors**: Check that GPU supports the required OptiX version
3. **Performance Issues**: Enable timing to identify bottlenecks
4. **Memory Issues**: Monitor GPU memory usage with nvidia-smi

### Debug Information

Enable debug output:
```bash
export BARNEY_DENOISER_TIMING=1  # Timing information
```

## References

- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/)
- [VisRTX Denoiser Implementation](https://github.com/NVIDIA/VisRTX)
- [CUDA Thrust Library](https://docs.nvidia.com/cuda/thrust/)
