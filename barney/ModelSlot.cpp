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

#include "barney/ModelSlot.h"
#include "barney/GlobalModel.h"
#include "barney/common/Data.h"
#include "barney/common/Texture.h"
#include "barney/light/Light.h"
#include "barney/geometry/Geometry.h"

namespace barney {

  ModelSlot::PLD *ModelSlot::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }

  ModelSlot::ModelSlot(GlobalModel *_model,
                       const DevGroup::SP &devices,
                       int slotID)
    : SlottedObject(_model->context,
                    devices),
      model(_model),
      slotID(slotID),
      slotContext(_model->context->getSlot(slotID)),
      world(std::make_shared<render::World>(slotContext))
  {
    perLogical.resize(devices->numLogical);
  }

  ModelSlot::~ModelSlot()
  {}
  
  void ModelSlot::setInstances(std::vector<Group::SP> &groups,
                               const affine3f *xfms)
  {
    int numUserInstances = (int)groups.size();
    instances.groups = groups;
    instances.xfms.resize(numUserInstances);
    std::copy(xfms,xfms+numUserInstances,instances.xfms.data());
    for (auto device : *devices) {
      device->sbtDirty = true;
      auto pld = getPLD(device);
      if (pld->instanceGroup) {
        // owlGroupRelease(instances.group);
        device->rtc->freeGroup(pld->instanceGroup);
        pld->instanceGroup = 0;
      }
    }
  }
  
  // Group *ModelSlot::createGroup(const std::vector<Geometry::SP> &geoms,
  //                               const std::vector<Volume::SP> &volumes)
  // {
  //   return getContext()->initReference
  //     (std::make_shared<Group>(this,geoms,volumes));
  // }

  // Volume *ModelSlot::createVolume(ScalarField::SP sf)
  // {
  //   return getContext()->initReference
  //     (std::make_shared<Volume>(devGroup.get(),sf));
  // }

  // ModelSlot::SP ModelSlot::create(GlobalModel *model, int localID)
  // {
  //   ModelSlot::SP slot = std::make_shared<ModelSlot>(model,localID);
  //   return slot;
  // }

  // render::HostMaterial::SP ModelSlot::getDefaultMaterial()
  // {
  //   if (!defaultMaterial) {
  //     defaultMaterial
  //       = render::HostMaterial::create(this,"AnariMatte");
  //     defaultMaterial->commit();
  //   }
  //   return defaultMaterial;
  // }
  
  // Data *ModelSlot::createData(BNDataType dataType,
  //                             size_t numItems,
  //                             const void *items)
  // {
  //   return getContext()->initReference(Data::create(this,dataType,numItems,items));
  // }
  
  // Light *ModelSlot::createLight(const std::string &type)
  // {
  //   return getContext()->initReference(Light::create(this,type));
  // }

  // Texture *Context::createTexture(BNTexelFormat texelFormat,
  //                                 int slot,
  //                                 vec2i size,
  //                                 const void *texels,
  //                                 BNTextureFilterMode  filterMode,
  //                                 BNTextureAddressMode addressMode,
  //                                 BNTextureColorSpace  colorSpace)
  // {
  //   return initReference(std::make_shared<Texture>
  //                        (this,slot,
  //                         texelFormat,size,texels,
  //                         filterMode,addressMode,colorSpace));
  // }

  void ModelSlot::build()
  {
    std::vector<affine3f> rtcTransforms;
    EnvMapLight::SP envMapLight;

    // ==================================================================
    // generate all lights's "raw" data. note this is NOT per device
    // (yet), even though the use of 'DD's seems to imply so. this
    // should "eventually" be changed to something where the current
    // 'world' class gets merged into 'modelslot', and all light,
    // material, and texture data then live 'per logical device'
    // ==================================================================
    std::vector<QuadLight::DD> quadLights;
    std::vector<DirLight::DD>  dirLights;
    std::pair<EnvMapLight::SP,affine3f> envLight;
    
    for (int i=0;i<instances.groups.size();i++) {
      Group *group = instances.groups[i].get();
      if (!group) continue;
      if (!group->lights) continue;
      for (auto &light : group->lights->items) {
        if (!light) continue;
        if (QuadLight::SP quadLight = light->as<QuadLight>()) {
          quadLights.push_back(quadLight->getDD(instances.xfms[i]));
          continue;
        } 
        if (DirLight::SP dirLight = light->as<DirLight>()) {
          dirLights.push_back(dirLight->getDD(instances.xfms[i]));
          continue;
        }
        if (EnvMapLight::SP el = light->as<EnvMapLight>()) {
          envLight = {el,instances.xfms[i]};
          continue;
        }
        throw std::runtime_error("un-handled type of light!?");
      }
    }
    world->set(envLight.first,envLight.second);
    world->set(quadLights);
    world->set(dirLights);
  
    // ==================================================================
    // generate all (per device) instance lists. note each BGGroup can
    // contain more than one rtcGroup, so theres's not a one-to-one
    // between barney instance list transform array and rtc instance
    // list transform array
    // ==================================================================
    
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      std::vector<affine3f>     rtcTransforms;
      std::vector<rtc::Group *> rtcGroups;
      
      for (int i=0;i<instances.groups.size();i++) {
        Group *group = instances.groups[i].get();
        if (!group) continue;
        Group::PLD *groupPLD = group->getPLD(device);
      
        if (groupPLD->userGeomGroup) {
          rtcGroups.push_back(groupPLD->userGeomGroup);
          rtcTransforms.push_back(instances.xfms[i]);
        }
        if (groupPLD->volumeGeomsGroup) {
          rtcGroups.push_back(groupPLD->volumeGeomsGroup);
          rtcTransforms.push_back(instances.xfms[i]);
        }
        if (groupPLD->triangleGeomGroup) {
          rtcGroups.push_back(groupPLD->triangleGeomGroup);
          rtcTransforms.push_back(instances.xfms[i]);
        }

        for (auto group : groupPLD->volumeGroups) {
          rtcGroups.push_back(group);
          rtcTransforms.push_back(instances.xfms[i]);
        }
      }
  
      if (pld->instanceGroup) {
        device->rtc->freeGroup(pld->instanceGroup);
        pld->instanceGroup = 0;
      }
      pld->instanceGroup
        = device->rtc->createInstanceGroup(rtcGroups,
                                           rtcTransforms);
      pld->instanceGroup->buildAccel();
    }
  }

}

  
