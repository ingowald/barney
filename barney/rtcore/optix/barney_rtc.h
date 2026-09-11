// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/common/rtcore-common.h"
#include "rtcore/cudaCommon/cuda-common.h"
#include "rtcore/optix/AppInterface.h"
#include "rtcore/optix/ComputeInterface.h"
#include "rtcore/optix/TraceInterface.h"

/*! tells the backend that this is the optix backend, so it can
    enable/disable stuff that should only be applicable to this
    backend (such as anari nv-framebuffer extensions */
#define BARNEY_RTC_OPTIX 1


inline void rtc_check()
{ BARNEY_CUDA_SYNC_CHECK(); }
