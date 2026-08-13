// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/common/rtcore-common.h"
#include "rtcore/cudaCommon/cuda-common.h"
#include "rtcore/cuda/AppInterface.h"
#include "rtcore/cudaCommon/ComputeInterface.h"
// only pipeline programs should ever see trace functions
#if BARNEY_DEVICE_PROGRAM
// #if RTC_DEVICE_CODE 
# include "rtcore/cuda/TraceInterface.h"
#endif



