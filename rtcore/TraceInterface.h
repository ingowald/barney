// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


/*! \file rtcore/TraceInterface.h \brief Defines the interface with
    which pipeline/device programs (closest hit, anythit, etc) can
    talk to to the ray tracing pipeline (eg, to ask for primitive ID,
    call ignoreIntersction(), etc */

#pragma once

#if BARNEY_RTC_OPTIX
# include "rtcore/optix/TraceInterface.h"
#endif

#if BARNEY_RTC_EMBREE
# include "rtcore/embree/TraceInterface.h"
#endif

#if BARNEY_RTC_CUDA
# include "rtcore/cuda/TraceInterface.h"
#endif

