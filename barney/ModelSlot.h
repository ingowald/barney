// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/Group.h"
#include "barney/material/Material.h"
#include "barney/render/World.h"
#include "barney/MultiPassObject.h"
#include <set>

namespace BARNEY_NS {

  struct GlobalModel;
  struct Context;
  struct Texture;
  struct Light;

  struct SlotContext;
  struct ModelSlot;
  
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
    void setInstanceAttributes(const std::string &which, const PODData::SP &data);

    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
      std::vector<int>       userIDs;
    } instances;

    std::vector<std::pair<MultiPassObject::SP,affine3f>> additionalPasses;

    rtc::AccelHandle getInstanceAccel(Device *device)
    {
      auto *pld = getPLD(device);
      if (!pld || !pld->instanceGroup)
        return 0;
      return rtc::getAccelHandle(pld->instanceGroup);
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
    render::World::SP    world;

  };
  
}
