// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/GlobalModel.h"
#include "barney/FromEnv.h"

namespace BARNEY_NS {
  namespace native {

    GlobalModel::GlobalModel(Context *context)
      : Object(context)
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
  
    void GlobalModel::render(Renderer *renderer,
                             Camera      *_camera,
                             FrameBuffer *_fb)
    {
      if (context->myRank() == 0 && FromEnv::get()->logQueues) 
        std::cout << "============================================ new frame\n";
      assert(context);
      FrameBuffer *fb = (FrameBuffer *)_fb;
      Camera *camera = (Camera *)_camera;
      assert(fb);
      context->ensureRayQueuesLargeEnoughFor(fb);
      context->render((Renderer*)renderer,this,camera,fb);
      if (profHook)
        profHook();
    }


    void GlobalModel::setInstances(int slot,
                                   Group **groups,
                                   const affine3f *xfms,
                                   int numInstances)
    { getSlot(slot)->setInstances(groups,xfms,numInstances); }

    void GlobalModel::updateInstanceTransforms(int slot,
                                               const affine3f *xfms,
                                               int numInstances)
    { getSlot(slot)->updateInstanceTransforms(xfms,numInstances); }

    void GlobalModel::setInstanceAttributes(int slot,
                                            const std::string &which,
                                            Data::SP data)
    { getSlot(slot)->setInstanceAttributes(which,data?data->as<PODData>():PODData::SP{}); }

    void GlobalModel::build(int slot)
    { getSlot(slot)->build(); }
      
    
  }
}
