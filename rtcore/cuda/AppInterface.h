// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Buffer.h"
#include "rtcore/cudaCommon/ComputeKernel.h"
#include "rtcore/cuda/TraceKernel.h"
// for setPrimCount etc
#include "rtcore/cuda/Geom.h"
// for buildAccel
#include "rtcore/cuda/Group.h"
// for createTexture
#include "rtcore/cudaCommon/TextureData.h"
// for getDD
#include "rtcore/cudaCommon/Texture.h"

namespace rtc {
  namespace cuda {

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
  extern ::rtc::cuda::GeomType *createGeomType_##typeName(::rtc::Device *);

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DD,has_ah,has_ch) \
  extern rtc::cuda::GeomType *createGeomType_##typeName(rtc::Device *);






  
