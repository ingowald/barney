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
#include "barney/render/HostMaterial.h"
// #include "barney/render/DeviceMaterial.h"

namespace barney {

  ModelSlot::ModelSlot(GlobalModel *model, int slot)
    : Object(model->context),
      model(model),
      localID(localID),
      devGroup(model->context->perSlot[slot].devGroup),
      world(model->context->perSlot[slot].devGroup.get())
  {}

  ModelSlot::~ModelSlot()
  {}

  OWLContext ModelSlot::getOWL() const
  {
    assert(devGroup);
    assert(devGroup->owl);
    return devGroup->owl;
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
      owlGroupRelease(instances.group);
      instances.group = 0;      
    }
  }
  
  Group *ModelSlot::createGroup(const std::vector<Geometry::SP> &geoms,
                                const std::vector<Volume::SP> &volumes)
  {
    return getContext()->initReference
      (std::make_shared<Group>(this,geoms,volumes));
  }

  Volume *ModelSlot::createVolume(ScalarField::SP sf)
  {
    return getContext()->initReference
      (std::make_shared<Volume>(devGroup.get(),sf));
  }


  // Cylinders *ModelSlot::createCylinders(const vec3f      *points,
  //                                       int               numPoints,
  //                                       const vec3f      *colors,
  //                                       bool              colorPerVertex,
  //                                       const vec2i      *indices,
  //                                       int               numIndices,
  //                                       const float      *radii,
  //                                       bool              radiusPerVertex,
  //                                       float             defaultRadius)
  // {
  //   return getContext()->initReference
  //     (std::make_shared<Cylinders>(this,
  //                                  points,numPoints,
  //                                  colors,colorPerVertex,
  //                                  indices,numIndices,
  //                                  radii,radiusPerVertex,defaultRadius));
  // }
    
  // Spheres *ModelSlot::createSpheres(// const Material &material,
  //                                   // const vec3f *origins,
  //                                   // int numOrigins,
  //                                   // const vec3f *colors,
  //                                   // const float *radii,
  //                                   // float defaultRadius
  //                                   )
  // {
  //   return getContext()->initReference
  //     (std::make_shared<Spheres>(this));
  // }

  Data *ModelSlot::createData(BNDataType dataType,
                              size_t numItems,
                              const void *items)
  {
    return getContext()->initReference(Data::create(this,dataType,numItems,items));
  }
  
  Light *ModelSlot::createLight(const std::string &type)
  {
    return getContext()->initReference(Light::create(this,type));
  }

  void ModelSlot::setRadiance(float radiance)
  {
    world.radiance = radiance;
  }
  
  Texture *ModelSlot::createTexture(BNTexelFormat texelFormat,
                                    vec2i size,
                                    const void *texels,
                                    BNTextureFilterMode  filterMode,
                                    BNTextureAddressMode addressMode,
                                    BNTextureColorSpace  colorSpace)
  {
    return getContext()->initReference
      (std::make_shared<Texture>(this,texelFormat,size,texels,
                                 filterMode,addressMode,colorSpace));
  }

  void ModelSlot::build()
  { 
    std::vector<render::QuadLight> quadLights;
    std::vector<render::DirLight>  dirLights;

    std::vector<affine3f> owlTransforms;
    std::vector<OWLGroup> owlGroups;
    EnvMapLight::SP envMapLight;
    
    for (int i=0;i<instances.groups.size();i++) {
      Group *group = instances.groups[i].get();
      if (group->lights)
        for (auto &light : group->lights->items) {
          if (!light) continue;
          if (QuadLight::SP quadLight = light->as<QuadLight>()) {
            quadLights.push_back(quadLight->content);
            continue;
          } 
          if (DirLight::SP dirLight = light->as<DirLight>()) {
            dirLights.push_back(dirLight->content);
            continue;
          }
          if (EnvMapLight::SP el = light->as<EnvMapLight>()) {
            envMapLight = el;
          }
        }
      
      if (group->userGeomGroup) {
        owlGroups.push_back(group->userGeomGroup);
        owlTransforms.push_back(instances.xfms[i]);
      }
      if (group->volumeGeomsGroup) {
        owlGroups.push_back(group->volumeGeomsGroup);
        owlTransforms.push_back(instances.xfms[i]);
      }
      if (group->triangleGeomGroup) {
        owlGroups.push_back(group->triangleGeomGroup);
        owlTransforms.push_back(instances.xfms[i]);
      }
      for (auto volume : group->volumes)
        for (auto gg : volume->generatedGroups) {
          owlGroups.push_back(gg);
          owlTransforms.push_back(instances.xfms[i]);
        }
    }
      
    // if (owlGroups.size() == 0)
    //   std::cout << OWL_TERMINAL_RED
    //             << "warning: data group is empty..."
    //             << OWL_TERMINAL_DEFAULT << std::endl;
    instances.group
      = owlInstanceGroupCreate(devGroup->owl,
                               owlGroups.size(),
                               owlGroups.data(),
                               nullptr,
                               (const float *)owlTransforms.data());
    owlGroupBuildAccel(instances.group);
    world.set(envMapLight?envMapLight->content:render::EnvMapLight{});
    world.set(quadLights);
    world.set(dirLights);
  }
    
}

  
