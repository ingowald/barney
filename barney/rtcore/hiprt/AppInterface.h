// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/hiprt/Device.h"
#include "rtcore/cuda/Buffer.h"
#include "rtcore/cudaCommon/ComputeKernel.h"
#include "rtcore/hiprt/TraceKernel.h"
#include "rtcore/cuda/Geom.h"
#include "rtcore/hiprt/Group.h"
#include "rtcore/cudaCommon/TextureData.h"
#include "rtcore/cudaCommon/Texture.h"

namespace rtc {
  namespace hiprt {

    using rtc::cuda_common::enablePeerAccess;
    using rtc::cuda_common::getPhysicalDeviceHash;

    using rtc::cuda_common::ComputeKernel1D;
    using rtc::cuda_common::ComputeKernel2D;
    using rtc::cuda_common::ComputeKernel3D;

    using rtc::cuda_common::Texture;
    using rtc::cuda_common::TextureData;
  }
}

#define RTC_IMPORT_USER_GEOM(moduleName,typeName,DD,has_ah,has_ch)      \
  extern ::rtc::hiprt::GeomType *createGeomType_##typeName(::rtc::Device *);

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern rtc::hiprt::GeomType *createGeomType_##typeName(rtc::Device *);
