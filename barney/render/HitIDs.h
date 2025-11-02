// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/math.h"

namespace BARNEY_NS {

  /*! HitIDs are used for a special "ID pass" that computes the
      closest intersection along a (primary) ray, WITHOUT any opacity
      or transparency taken into effect. As such it requires its own
      depth value */
  struct HitIDs {
    float depth = BARNEY_INF;
    int primID = -1;
    int instID = -1;
    int objID  = -1;
  };
  
} // ::BARNEY_NS
