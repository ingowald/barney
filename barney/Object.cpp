// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/Object.h"
#include "barney/Context.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {

  // Object::Object(Context *context)
  //   : context(context)
  // {}
  
  SlottedObject::SlottedObject(Context *context,
                               const DevGroup::SP &devices)
    : barney_api::Object(context),
      devices(devices)
  {
    assert(devices);
    assert(!devices->empty());
  }
 
}

