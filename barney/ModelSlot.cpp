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

  ModelSlot::ModelSlot(GlobalModel *_model, int slot)
    : Object(_model->context),
      model(_model),
      localID(slot),
      devGroup(_model->context->perSlot[slot].devGroup),
      world(std::make_shared<render::World>(_model->context->perSlot[slot].devGroup))
  {
  }

  ModelSlot::~ModelSlot()
  {}
  
  // OWLContext ModelSlot::getOWL() const
  // {
  //   assert(devGroup);
  //   assert(devGroup->owl);
  //   return devGroup->owl;
  // }
  rtc::DevGroup *ModelSlot::getRTC() const
  {
    assert(devGroup);
    assert(devGroup->rtc);
    return devGroup->rtc;
  }

  void ModelSlot::setInstances(std::vector<Group::SP> &groups,
                               const affine3f *xfms)
  {
    int numUserInstances = (int)groups.size();
    instances.groups = std::move(groups);
    instances.xfms.resize(numUserInstances);
    std::copy(xfms,xfms+numUserInstances,instances.xfms.data());
    devGroup->sbtDirty = true;
    if (instances.group) {
      // owlGroupRelease(instances.group);
      getRTC()->free(instances.group);
      instances.group = 0;      
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

  ModelSlot::SP ModelSlot::create(GlobalModel *model, int localID)
  {
    ModelSlot::SP slot = std::make_shared<ModelSlot>(model,localID);
    return slot;
  }

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
    std::vector<QuadLight::DD> quadLights;
    std::vector<DirLight::DD>  dirLights;

    // std::vector<affine3f> owlTransforms;
    std::vector<affine3f> rtcTransforms;
    // std::vector<OWLGroup> owlGroups;
    std::vector<rtc::Group *> rtcGroups;
    EnvMapLight::SP envMapLight;
    
    for (int i=0;i<instances.groups.size();i++) {
      Group *group = instances.groups[i].get();
      if (group->lights)
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
            envMapLight = el;
          }
        }
      
      if (group->userGeomGroup) {
        // owlGroups.push_back(group->userGeomGroup);
        rtcGroups.push_back(group->userGeomGroup);
        // owlTransforms.push_back(instances.xfms[i]);
        rtcTransforms.push_back(instances.xfms[i]);
      }
      if (group->volumeGeomsGroup) {
        // owlGroups.push_back(group->volumeGeomsGroup);
        rtcGroups.push_back(group->volumeGeomsGroup);
        // owlTransforms.push_back(instances.xfms[i]);
        rtcTransforms.push_back(instances.xfms[i]);
      }
      if (group->triangleGeomGroup) {
        // owlGroups.push_back(group->triangleGeomGroup);
        rtcGroups.push_back(group->triangleGeomGroup);
        // owlTransforms.push_back(instances.xfms[i]);
        rtcTransforms.push_back(instances.xfms[i]);
      }
      for (auto volume : group->volumes) {
        for (auto gg : volume->generatedGroups) {
          // owlGroups.push_back(gg);
          rtcGroups.push_back(gg);
          // owlTransforms.push_back(instances.xfms[i]);
          rtcTransforms.push_back(instances.xfms[i]);
        }
      }
    }
      
    // instances.group
    //   = owlInstanceGroupCreate(devGroup->owl,
    //                            owlGroups.size(),
    //                            owlGroups.data(),
    //                            nullptr,
    //                            (const float *)owlTransforms.data());
    instances.group
      = getRTC()->createInstanceGroup(rtcGroups,
                                      rtcTransforms);
    // owlGroupBuildAccel(instances.group);
    instances.group->buildAccel();
    world->set(envMapLight);
    world->set(quadLights);
    world->set(dirLights);
  }
    
}

  
