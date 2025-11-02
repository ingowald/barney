// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/optix/Device.h"
#include "rtcore/optix/Buffer.h"
#include "rtcore/cudaCommon/ComputeKernel.h"
// for setPrimCount etc
#include "rtcore/optix/Geom.h"
// for buildAccel
#include "rtcore/optix/Group.h"
// for createTexture
#include "rtcore/cudaCommon/TextureData.h"
// for getDD
#include "rtcore/cudaCommon/Texture.h"
#include "rtcore/optix/Denoiser.h"

namespace rtc {
  namespace optix {

    using rtc::cuda_common::enablePeerAccess;
    using rtc::cuda_common::getPhysicalDeviceHash;
    
    using rtc::cuda_common::ComputeKernel1D;
    using rtc::cuda_common::ComputeKernel2D;
    using rtc::cuda_common::ComputeKernel3D;
    
    using rtc::cuda_common::Texture;
    using rtc::cuda_common::TextureData;

  }
}

