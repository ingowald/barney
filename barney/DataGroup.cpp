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
#include "barney/Spheres.h"
#include "barney/Triangles.h"

namespace barney {

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
    if (instances.group) {
      owlGroupRelease(instances.group);
      instances.group = 0;      
    }
  }
  
  Group   *DataGroup::createGroup(const std::vector<Geometry::SP> &geoms)
  {
    assert(model);
    assert(model->context);
    return model->context->initReference
      (std::make_shared<Group>(this,geoms));
  }

  Spheres *DataGroup::createSpheres(const Material &material,
                                    const vec3f *origins,
                                    int numOrigins,
                                    const float *radii,
                                    float defaultRadius)
  {
    assert(model);
    assert(model->context);
    return model->context->initReference
      (std::make_shared<Spheres>(this,material,origins,numOrigins,radii,defaultRadius));
  }
  
  
  void DataGroup::build()
  {
    std::vector<affine3f> owlTransforms;
    std::vector<OWLGroup> owlGroups;
    for (int i=0;i<instances.groups.size();i++) {
      Group *group = instances.groups[i].get();
        
      if (group->userGeomGroup) {
        owlGroups.push_back(group->userGeomGroup);
        owlTransforms.push_back(instances.xfms[i]);
      }
      if (group->triangleGeomGroup) {
        owlGroups.push_back(group->triangleGeomGroup);
        owlTransforms.push_back(instances.xfms[i]);
      }
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

  