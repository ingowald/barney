// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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

  void (*profHook)() = nullptr;
  
  void GlobalModel::render(barney_api::Renderer *renderer,
                           barney_api::Camera      *_camera,
                           barney_api::FrameBuffer *_fb)
  {
    auto _context = (BARNEY_NS::Context *)this->context;
    if (context->myRank() == 0 && FromEnv::get()->logQueues) 
      std::cout << "============================================ new frame\n";
    assert(context);
    FrameBuffer *fb = (FrameBuffer *)_fb;
    Camera *camera = (Camera *)_camera;
    assert(fb);
    Context *context = (Context *)this->context;
    context->ensureRayQueuesLargeEnoughFor(fb);
    context->render((Renderer*)renderer,this,camera,fb);
    if (profHook)
      profHook();
  }

}
