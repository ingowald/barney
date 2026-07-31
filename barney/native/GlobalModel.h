// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Context.h"
#include "native/ModelSlot.h"

namespace BARNEY_NS {
  namespace native {

    struct GlobalModel : public Object {
      typedef std::shared_ptr<GlobalModel> SP;

      static SP create(Context *ctx) { return std::make_shared<GlobalModel>(ctx); }
    
      GlobalModel(Context *context);
      virtual ~GlobalModel();
    
      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "Model{}"; }
      
      void setInstances(int slot,
                        Group **groups,
                        const affine3f *xfms,
                        int numInstances);
      void updateInstanceTransforms(int slot,
                                    const affine3f *xfms,
                                    int numInstances);
      void setInstanceAttributes(int slot,
                                 const std::string &which,
                                 Data::SP data);
      void build(int slot);
      void render(Renderer    *renderer,
                  Camera      *camera,
                  FrameBuffer *fb);

      ModelSlot *getSlot(int whichSlot)
      {
        assert(whichSlot >= 0);
        assert(whichSlot < modelSlots.size());
        return modelSlots[whichSlot].get();
      }
      std::vector<ModelSlot::SP> modelSlots;
    };
    
  }
} // ::BARNEY_NS
