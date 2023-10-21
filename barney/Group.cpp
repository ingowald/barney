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

namespace barney {
  Group::Group(DataGroup *owner,
               const std::vector<Geometry::SP> &geoms,
               const std::vector<Volume::SP> &volumes)
    : owner(owner),
      geoms(geoms),
      volumes(volumes)
  {
  }
  
  void Group::build()
  {
    userGeoms.clear();
    triangleGeoms.clear();
    
    if (triangleGeomGroup)
      owlGroupRelease(triangleGeomGroup);
    if (userGeomGroup)
      owlGroupRelease(userGeomGroup);
    
    for (auto geom : geoms) {
      geom->build();
      for (auto g : geom->triangleGeoms)
        triangleGeoms.push_back(g);
      for (auto g : geom->userGeoms)
        userGeoms.push_back(g);
    }
    for (auto volume : volumes) {
      volume->build();
      for (auto g : volume->triangleGeoms)
        triangleGeoms.push_back(g);
      for (auto g : volume->userGeoms)
        userGeoms.push_back(g);
    }
    if (!userGeoms.empty())
      userGeomGroup = owlUserGeomGroupCreate
        (owner->devGroup->owl,userGeoms.size(),userGeoms.data());
    if (userGeomGroup) {
      std::cout << "building USER group" << std::endl;
      owlGroupBuildAccel(userGeomGroup);
    }

    if (!triangleGeoms.empty())
      triangleGeomGroup = owlTrianglesGeomGroupCreate
        (owner->devGroup->owl,triangleGeoms.size(),triangleGeoms.data());
    if (triangleGeomGroup) {
      std::cout << "building TRIANGLES group" << std::endl;
      owlGroupBuildAccel(triangleGeomGroup);
    }
  }
  
}
