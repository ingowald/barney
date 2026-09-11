// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// auto-generated config file that passes cmake variables
#include "barneyDeviceConfig.h"

#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
// #include <mutex>
#include <vector>
// #include <map>
// #include <memory>
#include <sstream>

# if defined(__CUDA_ARCH__) || defined(__HIPCC__)
// # if defined(__CUDACC__) || defined(__HIPCC__)
// # ifdef __CUDA_ARCH__

// let's the barney source files know whether they're currently being
// comiled for 'device' execution, or not. For cuda etc, this means
// we're in the __CUDA_ARCH__ pass, for cpu this is going to be turned
// on always.
#  define RTC_DEVICE_CODE 1
# endif

#include <owl/common/owl-common.h>
#include <owl/common/math/box.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>

#define __barney_align(a) OWL_ALIGN(a)

namespace BARNEY_NS {
  namespace rtc {
  
    using namespace owl::common;
    
    using range1f = interval<float>;
  
    typedef enum {
      UCHAR=0,
      UCHAR4,

      INT=10,
      INT2,
      INT3,
      INT4,
      
      LONG=20,
      LONG2,
      LONG3,
      LONG4,
      
      FLOAT=30,
      FLOAT2,
      FLOAT3,
      FLOAT4,
      
      USHORT=40,
      UCHAR2=41,
      USHORT2=42,
      HALF=50,
      HALF2=51,
      HALF3=52,
      HALF4=53,
    } DataType;

    typedef enum {
      WRAP,CLAMP,BORDER,MIRROR,
    } AddressMode;
      
    typedef enum {
      FILTER_MODE_POINT,FILTER_MODE_LINEAR,
    } FilterMode;
    
    typedef enum {
      COLOR_SPACE_LINEAR, COLOR_SPACE_SRGB,
    } ColorSpace;

    // IEEE 754 half decode - the only half support we need: plain
    // bit math, portable to every backend and to device code (no
    // CUDA headers required). inf/nan are built from ldexpf since
    // std::numeric_limits is not device-callable.
    inline __both__ float halfToFloat(uint16_t h)
    {
      const uint32_t sign = (uint32_t)(h >> 15);
      const uint32_t exp  = (uint32_t)((h >> 10) & 0x1f);
      const uint32_t man  = (uint32_t)(h & 0x3ff);
      float v;
      if (exp == 0)
        v = ldexpf((float)man, -24);
      else if (exp == 0x1f)
        v = (man == 0u) ? ldexpf(1.f, 128)          // inf
                        : ldexpf(1.f, 128) - ldexpf(1.f, 128); // inf-inf = nan
      else
        v = ldexpf((float)(man + 1024u), (int)exp - 25);
      return sign ? -v : v;
    }
  
    struct TextureDesc {
      FilterMode filterMode = FILTER_MODE_LINEAR;
      AddressMode addressMode[3] = { CLAMP, CLAMP, CLAMP };
      vec4f borderColor          = {0.f,0.f,0.f,0.f};
      bool normalizedCoords      = true;
      ColorSpace colorSpace      = COLOR_SPACE_LINEAR;
    };

    typedef struct _TextureObject *TextureObject;
    typedef struct _AccelHandle   *AccelHandle;
  }
}



