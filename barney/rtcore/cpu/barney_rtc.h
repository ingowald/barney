// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cpu/AppInterface.h"
#define BARNEY_DEVICE_PROGRAM 1
#include "rtcore/cpu/TraceInterface.h"


/*! tells the barney/native/ and barney/anari/ components that use
    this backend which kind of backend it is */
#define BARNEY_RTC_CUDA 1
