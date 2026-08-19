// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"

namespace BARNEY_NS {
  namespace native {
    
    struct FromEnv {
      static bool enabled(const std::string &key);
      
      /*! allows for querying whether a value _was_ set _and_ set to
        false. E.g, 'denoising=0' will return true for
        explicitDisabled("denosing"); "denoising=1' would return false
        (because it's _en_abled, not disabled), and 'denoising' not
        set at all would return false (because it hasn't even been
        set, and thus not explicitly disabled */
      static bool explicitlyDisabled(const std::string &key);
      
      static void init();
      
      static bool logQueues;
      static bool skipDenoising;
      static bool logConfig;
      static bool logBackend;
      static bool logTopo;
    };

  }
}
