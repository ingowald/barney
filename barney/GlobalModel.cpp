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

namespace BARNEY_NS {

  GlobalModel::GlobalModel(Context *context)
    : SlottedObject(context,context->devices)
  {
    for (int slot=0;slot<context->perSlot.size();slot++) {
      assert(context->perSlot[slot].devices);
      ModelSlot::SP modelSlot
        = std::make_shared<ModelSlot>(this,context->perSlot[slot].devices,
                                      slot);
      modelSlots.push_back(modelSlot);
    }
  }

  GlobalModel::~GlobalModel()
  {}
  
  void GlobalModel::render(Renderer *renderer,
                           Camera      *camera,
                           FrameBuffer *fb)
  {
    assert(context);
    assert(fb);
    context->ensureRayQueuesLargeEnoughFor(fb);
    context->render(renderer,this,camera->getDD(),fb);
  }

}
