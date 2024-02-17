// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/Camera.h"

namespace barney {

  Camera::Camera(Context *owner)
    : Object(owner)
  {}
  
  Camera::SP Camera::create(Context *owner,
                            const char *type)
  {
    // iw - "eventually" we should have different cameras like
    // 'perspective' etc here, but for now, let's just
    // ignore the type and create a single one thta contains all
    // fields....
    return std::make_shared<Camera>(owner);
  }

}

