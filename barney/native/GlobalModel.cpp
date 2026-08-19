// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/GlobalModel.h"
#include "native/FromEnv.h"

namespace BARNEY_NS {
  namespace native {

    GlobalModel::GlobalModel(Context *context)
      : Object(context)
    {
      for (auto &ldg : context->perLDG)
        modelSlots.push_back
          (std::make_shared<ModelSlot>(this,ldg));
    }

    GlobalModel::~GlobalModel()
    {}

    void (*profHook)() = nullptr;
  
    void GlobalModel::render(Renderer *renderer,
                             Camera      *_camera,
                             FrameBuffer *_fb)
    {
      if (context->myRank() == 0 && FromEnv::logQueues) 
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
