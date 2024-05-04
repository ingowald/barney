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

#include "barney/GlobalModel.h"

namespace barney {

  GlobalModel::GlobalModel(Context *context)
    : Object(context)
  {
    PING;
    for (int slot=0;slot<context->perSlot.size();slot++) {
      PING; PRINT(this);
      ModelSlot::SP modelSlot = ModelSlot::create(this,slot);
      PRINT((int*)modelSlot.get());
      modelSlots.push_back(modelSlot);
      PING;
    }
    PING;
  }

  GlobalModel::~GlobalModel()
  {}
  
  void GlobalModel::render(Camera      *camera,
                           FrameBuffer *fb,
                           int pathsPerPixel)
  {
    assert(context);
    assert(fb);
    context->ensureRayQueuesLargeEnoughFor(fb);
    context->render(this,camera->getDD(),fb,pathsPerPixel);
  }

}
