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

#include "barney/LocalFB.h"
#include "barney/Model.h"
#include "barney/Spheres.h"
#include "owl/owl.h"

namespace barney {

  Model::Model(Context *context)
    : context(context)
  {
    for (int localID=0;localID<context->dataGroupIDs.size();localID++) {
      dataGroups.push_back(DataGroup::create(this,localID));
    }
  }

  DataGroup::DataGroup(Model *model, int localID)
    : model(model),
      localID(localID),
      devGroup(model->context->perDG[localID].devGroup)
  {}

  Group::Group(DataGroup *owner,
               const std::vector<Geom::SP> &geoms)
    : geoms(geoms)
  {
  }
  
  void Group::build()
  {
  }
  
  Group   *DataGroup::createGroup(const std::vector<Geom::SP> &geoms)
  {
    assert(model);
    assert(model->context);
    return model->context->initReference(Group::create(this,geoms));
  }

  Spheres *DataGroup::createSpheres(const mori::Material &material,
                                    const vec3f *origins,
                                    int numOrigins,
                                    const float *radii,
                                    float defaultRadius)
  {
    assert(model);
    assert(model->context);
    return model->context->initReference
      (Spheres::create(this,material,origins,numOrigins,radii,defaultRadius));
  }
  
  
  void DataGroup::build()
  {
    std::set<Group::SP> groups;
    for (auto group : instances.groups)
      groups.insert(group);

    for (auto group : groups)
      group->build();
  }
  
  
  void Model::render(const mori::Camera *camera,
                     FrameBuffer *fb)
  {
    assert(context);
    assert(fb);
    assert(camera);
    context->ensureRayQueuesLargeEnoughFor(fb);
    context->render(this,camera,fb);
  }

}
