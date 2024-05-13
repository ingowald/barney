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
#include "barney/render/HostMaterial.h"
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
    virtual ~ModelSlot() = default;

    render::HostMaterial::SP getDefaultMaterial()
    {
      if (!defaultMaterial)
        defaultMaterial
          = render::HostMaterial::create(this,"AnariMatte");
      return defaultMaterial;
    }
    render::HostMaterial::SP defaultMaterial = 0;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "barney::ModelSlot"; }
    
    OWLContext getOWL() const;

    static SP create(GlobalModel *model, int localID);

    Group   *
    createGroup(const std::vector<Geometry::SP> &geoms,
                const std::vector<Volume::SP> &volumes);
    
    Volume *createVolume(ScalarField::SP sf);
    
    Texture *
    createTexture(BNTexelFormat texelFormat,
                  vec2i size,
                  const void *texels,
                  BNTextureFilterMode  filterMode,
                  BNTextureAddressMode addressMode,
                  BNTextureColorSpace  colorSpace);
    
    ScalarField *createBlockStructuredAMR(std::vector<box3i> &blockBounds,
                                          std::vector<int> &blockLevels,
                                          std::vector<int> &blockOffsets,
                                          std::vector<float> &blockScalars);

    Data *createData(BNDataType dataType,
                     size_t numItems,
                     const void *items);
    Light *createLight(const std::string &type);
    void setRadiance(float radiance);
    void setInstances(std::vector<Group::SP> &groups,
                      const affine3f *xfms);
    
    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
      OWLGroup group = 0;
    } instances;

    void build();

    render::World world;
    // MultiPass::Instances multiPassInstances;
    DevGroup::SP   const devGroup;
    GlobalModel   *const model;
    int            const localID;
  };
  
}
