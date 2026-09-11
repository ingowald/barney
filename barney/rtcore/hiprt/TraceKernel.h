// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace hiprt {

    struct Device;

    typedef void (*TraceLaunchFct)(Device *device, vec2i dims, const void *lpData);

    struct TraceKernel2D {
      TraceKernel2D(Device *device,
                    size_t sizeOfLP,
                    TraceLaunchFct traceLaunchFct);
      ~TraceKernel2D();
      void launch(vec2i launchDims, const void *kernelData);

      Device        *const device;
      TraceLaunchFct const traceLaunchFct;
      size_t         const sizeOfLP;
      void          *d_lpData = 0;
    };

  }
}

#define RTC_IMPORT_TRACE2D(fileNameBase,name,sizeOfLP)                  \
  void rtc_hiprt_launch_##name(rtc::Device *device,                    \
                               vec2i dims,                              \
                               const void *lpData);                     \
                                                                        \
  ::rtc::TraceKernel2D *createTrace_##name(rtc::Device *device)         \
  {                                                                     \
    return new ::rtc::hiprt::TraceKernel2D                              \
      (device,sizeOfLP,rtc_hiprt_launch_##name);                        \
  }
