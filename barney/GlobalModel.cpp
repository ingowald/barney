// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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
    : barney_api::Model(context)
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
  
  void GlobalModel::render(barney_api::Renderer *renderer,
                           barney_api::Camera      *_camera,
                           barney_api::FrameBuffer *_fb)
  {
    if (context->myRank() == 0)
      std::cout << "============================================ new frame\n";
    assert(context);
    FrameBuffer *fb = (FrameBuffer *)_fb;
    Camera *camera = (Camera *)_camera;
    assert(fb);
    Context *context = (Context *)this->context;
    context->ensureRayQueuesLargeEnoughFor(fb);
    context->render((Renderer*)renderer,this,camera,fb);
  }

}
