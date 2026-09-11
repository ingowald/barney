// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/globalTrace/RQSBase.h"

namespace BARNEY_NS {
  namespace native {

    /*! implements queue cycling for a local context only, where we can
      use memcpy to move rays between different local devices */
    struct RQSLocal : public RQSBase
    {
      RQSLocal(Context *context);
    
      /*! forward rays (during global trace); returns if _after_ that
        forward the rays need more tracing (true) or whether they're
        done (false) */
      bool forwardRays(bool needHitIDs) override;
    };

  }
}

