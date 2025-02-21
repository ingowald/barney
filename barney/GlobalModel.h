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

#include "barney/Context.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {

  struct ModelSlot;
  
  struct GlobalModel : public SlottedObject {
    typedef std::shared_ptr<GlobalModel> SP;

    static SP create(Context *ctx) { return std::make_shared<GlobalModel>(ctx); }
    
    GlobalModel(Context *context);
    virtual ~GlobalModel();
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Model{}"; }

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
