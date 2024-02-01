// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/DataGroup.h"
#include "barney/Model.h"
#include "barney/Texture.h"
#include "barney/geometry/Spheres.h"
#include "barney/geometry/Cylinders.h"
#include "barney/geometry/Triangles.h"

namespace barney {
  
  Context *DataGroup::getContext() const
  {
    assert(model);
    assert(model->context);
    return model->context;
  }

  OWLContext DataGroup::getOWL() const
  {
    assert(devGroup);
    assert(devGroup->owl);
    return devGroup->owl;
  }

  DataGroup::DataGroup(Model *model, int localID)
    : model(model),
      localID(localID),
      devGroup(model->context->perDG[localID].devGroup)
  {}

  void DataGroup::setInstances(std::vector<Group::SP> &groups,
                               const affine3f *xfms)
  {
    int numUserInstances = groups.size();
    instances.groups = std::move(groups);
    instances.xfms.resize(numUserInstances);
    std::copy(xfms,xfms+numUserInstances,instances.xfms.data());
    devGroup->sbtDirty = true;
    if (instances.group) {
      owlGroupRelease(instances.group);
      instances.group = 0;      
    }
  }
  
  Group *DataGroup::createGroup(const std::vector<Geometry::SP> &geoms,
                                const std::vector<Volume::SP> &volumes)
  {
    return getContext()->initReference
      (std::make_shared<Group>(this,geoms,volumes));
  }

  Volume *DataGroup::createVolume(ScalarField::SP sf)
  {
    return getContext()->initReference
      (std::make_shared<Volume>(devGroup.get(),sf));
  }


  Cylinders *DataGroup::createCylinders(const Material   &material,
                                        const vec3f      *points,
                                        int               numPoints,
                                        const vec2i      *indices,
                                        int               numIndices,
                                        const float      *radii,
                                        float             defaultRadius)
  {
    return getContext()->initReference
      (std::make_shared<Cylinders>(this,material,
                                   points,numPoints,
                                   indices,numIndices,
                                   radii,defaultRadius));
  }
    
  Spheres *DataGroup::createSpheres(const Material &material,
                                    const vec3f *origins,
                                    int numOrigins,
                                    const vec3f *colors,
                                    const float *radii,
                                    float defaultRadius)
  {
    return getContext()->initReference
      (std::make_shared<Spheres>(this,material,origins,numOrigins,colors,radii,defaultRadius));
  }

  Texture *DataGroup::createTexture(BNTexelFormat texelFormat,
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
    
  Triangles *DataGroup::createTriangles(const barney::Material &material,
                                        int numIndices,
                                        const vec3i *indices,
                                        int numVertices,
                                        const vec3f *vertices,
                                        const vec3f *normals,
                                        const vec2f *texcoords)
  {
    return getContext()->initReference
      (std::make_shared<Triangles>(this,material,
                                   numIndices,
                                   indices,
                                   numVertices,
                                   vertices,
                                   normals,
                                   texcoords));
  }

  void DataGroup::build()
  { 
    multiPassInstances.clear();
   
    std::vector<affine3f> owlTransforms;
    std::vector<OWLGroup> owlGroups;
    for (int i=0;i<instances.groups.size();i++) {
      Group *group = instances.groups[i].get();
        
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
      multiPassInstances.instantiate(group,instances.xfms[i]);
    }

    if (owlGroups.size() == 0)
      std::cout << OWL_TERMINAL_RED
                << "warning: data group is empty..."
                << OWL_TERMINAL_DEFAULT << std::endl;
    instances.group
      = owlInstanceGroupCreate(devGroup->owl,
                               owlGroups.size(),
                               owlGroups.data(),
                               nullptr,
                               (const float *)owlTransforms.data());
    owlGroupBuildAccel(instances.group);
  }

}

  
