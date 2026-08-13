// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tell whoever includes us that we're the CPU backend
#define BARNEY_RTC_CPU 1

#include "rtcore/cpu/AppInterface.h"
#define BARNEY_DEVICE_PROGRAM 1
#include "rtcore/cpu/TraceInterface.h"

