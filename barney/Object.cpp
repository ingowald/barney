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

