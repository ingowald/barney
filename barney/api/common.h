// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/barneyConfig.h"
#include <cstring>
#include <cassert>
#include <mutex>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include "barney/barney.h"
// #include "barney/api/mat4.h"

#include "owl/common/math/AffineSpace.h"

namespace barney_api {
  using namespace owl::common;

  typedef owl::common::interval<float> range1f;
}
