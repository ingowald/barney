// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
  struct SlottedObject : public barney_api::ParameterizedObject {
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
