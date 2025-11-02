// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"
#include "api/Context.h"
#include "barney/DeviceGroup.h"

namespace BARNEY_NS {

  struct Context;
  struct ModelSlot;
  using barney_api::Data;
  
  namespace render {
    struct World;
  };

  /*! a object owned (only) in a particular data group */
  struct SlottedObject : public barney_api::Object {
    SlottedObject(Context *context, const DevGroup::SP &devices);
    virtual ~SlottedObject() = default;

    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "<SlottedObject>"; }

    const DevGroup::SP devices;    
  };

  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
}
