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
#include "barney.h"
#include <set>

namespace barney {

  struct GlobalModel;
  struct Context;
  struct Texture;
  struct Data;
  struct Light;
  

  struct ModelSlot : public Object {
    typedef std::shared_ptr<ModelSlot> SP;

    ModelSlot(GlobalModel *model,
              /*! index with which the given rank's context will refer
                  to this _locally_; not the data rank in it */
              int slotIndex);
    virtual ~ModelSlot();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "barney::ModelSlot"; }
    
    OWLContext getOWL() const;

    static SP create(GlobalModel *model, int localID);

    void setInstances(std::vector<Group::SP> &groups,
                      const affine3f *xfms);
    
    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
      OWLGroup group = 0;
    } instances;

    void build();

    render::World::SP world;
    // MultiPass::Instances multiPassInstances;
    DevGroup::SP   const devGroup;
    GlobalModel   *const model;
    int            const localID;

  };
  
}
