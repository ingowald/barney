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

#include "DenoiserUtils.h"
#include <cuda_runtime.h>

// Use the common CUDA header that includes proper type definitions
#include "rtcore/cudaCommon/cuda-common.h"

namespace rtc {
  namespace optix {

    __device__ float linear_to_srgb_device(float x) {
      return (x <= 0.0031308f) ? 12.92f * x : 1.055f * powf(x, 1.f/2.4f) - 0.055f;
    }

    __device__ uint32_t pack_rgba(float r, float g, float b, float a) {
      uint32_t ri = min(255, max(0, int(r * 256.f)));
      uint32_t gi = min(255, max(0, int(g * 256.f)));
      uint32_t bi = min(255, max(0, int(b * 256.f)));
      uint32_t ai = min(255, max(0, int(a * 256.f)));
      return (ri << 0) | (gi << 8) | (bi << 16) | (ai << 24);
    }

    __global__ void convert_float4_to_rgba_kernel(
        const float4* input,
        uint32_t* output,
        int width,
        int height,
        bool srgb)
    {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int idy = blockIdx.y * blockDim.y + threadIdx.y;
      
      if (idx >= width || idy >= height) return;
      
      int pixel_idx = idy * width + idx;
      float4 pixel = input[pixel_idx];
      
      if (srgb) {
        // Convert linear to sRGB
        pixel.x = linear_to_srgb_device(pixel.x);
        pixel.y = linear_to_srgb_device(pixel.y);
        pixel.z = linear_to_srgb_device(pixel.z);
      }
      
      output[pixel_idx] = pack_rgba(pixel.x, pixel.y, pixel.z, pixel.w);
    }

    void convert_float4_to_rgba(
        const void* input,
        void* output,
        int width,
        int height,
        bool srgb,
        cudaStream_t stream)
    {
      dim3 blockSize(16, 16);
      dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                    (height + blockSize.y - 1) / blockSize.y);
      
      convert_float4_to_rgba_kernel<<<gridSize, blockSize, 0, stream>>>(
          (const float4*)input,
          (uint32_t*)output,
          width,
          height,
          srgb);
    }

  } // namespace optix
} // namespace rtc
