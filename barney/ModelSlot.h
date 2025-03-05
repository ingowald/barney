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

#include "barney/Group.h"
#include "barney/material/Material.h"
#include "barney/render/World.h"
#include <set>

namespace BARNEY_NS {

  struct GlobalModel;
  struct Context;
  struct Texture;
  struct Light;

  struct SlotContext;

  struct ModelSlot : public SlottedObject {
    typedef std::shared_ptr<ModelSlot> SP;

    ModelSlot(GlobalModel *model,
              const DevGroup::SP &devices,
              /*! index with which the given rank's context will refer
                  to this _locally_; not the data rank in it */
              int slotID);
    virtual ~ModelSlot();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "barney::ModelSlot"; }
    
    void setInstances(barney_api::Group **groups,
                      const affine3f *xfms,
                      int numInstances);

    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
    } instances;

    rtc::device::AccelHandle getInstanceAccel(Device *device)
    {
      auto *pld = getPLD(device);
      if (!pld || !pld->instanceGroup)
        return 0;
      return pld->instanceGroup->getDD();
    }
    struct PLD {
      rtc::Group *instanceGroup = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;

    void build();

    // ------------------------------------------------------------------
    // do not change order of these:
    // ------------------------------------------------------------------
    int            const slotID;
    GlobalModel   *const model;
    SlotContext   *const slotContext;
    render::World::SP world;

  };
  
}
