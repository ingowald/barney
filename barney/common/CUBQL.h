// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"
#include "cuBQL/bvh.h"
#if BARNEY_CUBQL_HOST
# include "cuBQL/builder/cpu.h"
#else
# include "cuBQL/builder/cuda.h"
#endif
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

// #ifdef __CUDACC__
// namespace cuBQL {
//   using float3 = ::float3;
//   using float4 = ::float4;
// }
// #endif

namespace BARNEY_NS {
  
  inline __both__ vec3f to_barney(cuBQL::vec3f v)
  { return vec3f(v.x,v.y,v.z); }
  
  inline __both__ cuBQL::vec3f to_cubql(vec3f v)
  { return {v.x,v.y,v.z}; }
  
} // ::barney
