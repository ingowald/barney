// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/*! \file rtcore/AppInterface.h Defines how the app/host side of
    barney can talk to rtcore; e.g., to create new groups, geometries,
    etc */

#pragma once

#if BARNEY_RTC_CUDA
# include "cuda/AppInterface.h"
namespace rtc { using namespace rtc::cuda; }
#endif

#if BARNEY_RTC_OPTIX
# include "optix/AppInterface.h"
namespace rtc { using namespace rtc::optix; }
#endif

#if BARNEY_RTC_EMBREE
# include "embree/AppInterface.h"
namespace rtc { using namespace rtc::embree; }
#endif
