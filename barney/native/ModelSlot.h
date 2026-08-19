// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Group.h"
#include "native/material/HostMaterial.h"
#include "native/render/World.h"
#include <set>

namespace BARNEY_NS {
  namespace native {
    
    struct GlobalModel;
    struct Context;
    struct Texture;
    struct Light;

    struct LDGContext;

    /* iw - TODO merge this with LDGContext - 'old' native barney had
       the concept of having multiple models, but anari doesn't have
       that (it's closest equivalent is multiple 'frame' objects), so
       this does't make any sense to keep around any more */
    struct ModelSlot : public SlottedObject {
      typedef std::shared_ptr<ModelSlot> SP;

      ModelSlot(GlobalModel *model,
                LDGContext *ldgContext);
                // const DevGroup::SP &devices
                // // ,
                // // /*! index with which the given rank's context will refer
                // //   to this _locally_; not the data rank in it */
                // // int slotID
                // );
      virtual ~ModelSlot();

      /*! pretty-printer for printf-debugging */
      std::string toString() const override { return "barney::ModelSlot"; }
    
      void setInstances(Group **groups,
                        const affine3f *xfms,
                        int numInstances);
      void updateInstanceTransforms(const affine3f *xfms, int numInstances);
      void setInstanceAttributes(const std::string &which, const PODData::SP &data);
      void updateWorldLightsFromInstances();
      void flattenInstancesForDevice(Device *device,
                                     std::vector<rtc::Group *> *rtcGroups,
                                     std::vector<affine3f> &rtcTransforms,
                                     std::vector<int> *inputInstIDs);

      struct {
        std::vector<Group::SP> groups;
        std::vector<affine3f>  xfms;
        std::vector<int>       userIDs;
      } instances;

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
      // int            const slotID;
      GlobalModel   *const model;
      // SlotContext   *const slotContext;
      LDGContext    *const ldgContext;
      World::SP    world;

    };

  }
}
