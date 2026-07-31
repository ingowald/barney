// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"
#include "cuBQL/bvh.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

namespace BARNEY_NS {
  namespace native {
    
    inline __rtc_both vec3f to_barney(cuBQL::vec3f v)
    { return vec3f(v.x,v.y,v.z); }
  
    inline __rtc_both cuBQL::vec3f to_cubql(vec3f v)
    { return {v.x,v.y,v.z}; }

  }
} // ::barney
