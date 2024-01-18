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
  {}
  
  /*! pretty-printer for printf-debugging */
  std::string Group::toString() const 
  { return "Group{}"; }
  
  void Group::build()
  {
    // triangles and user geoms - for now always rebuild
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
      if (!userGeoms.empty())
        userGeomGroup = owlUserGeomGroupCreate
          (owner->devGroup->owl,userGeoms.size(),userGeoms.data());
      if (userGeomGroup) {
        std::cout << "building USER GEOM group" << std::endl;
        owlGroupBuildAccel(userGeomGroup);
      }
      
      if (!triangleGeoms.empty())
        triangleGeomGroup = owlTrianglesGeomGroupCreate
          (owner->devGroup->owl,triangleGeoms.size(),triangleGeoms.data());
      if (triangleGeomGroup) {
        std::cout << "building TRIANGLES GEOM group" << std::endl;
        owlGroupBuildAccel(triangleGeomGroup);
      }
    }

    // volumes - these may need two passes
    {
      bool needRefit = false;
      bool needRebuild = false;
      std::vector<Volume *> ownGroupVolumes;
      std::vector<Volume *> mergedGroupVolumes;
      for (auto &volume : volumes) {
        switch (volume->updateMode()) {
        case VolumeAccel::FULL_REBUILD:
          mergedGroupVolumes.push_back(volume.get());
          needRebuild = true;
          break;
        case VolumeAccel::REFIT:
          mergedGroupVolumes.push_back(volume.get());
          needRefit = true;
          break;
        case VolumeAccel::BUILD_THEN_REFIT:
          mergedGroupVolumes.push_back(volume.get());
          needRebuild = true;
          needRefit = true;
          break;
        case VolumeAccel::HAS_ITS_OWN_GROUP:
          ownGroupVolumes.push_back(volume.get());
          break;
        default:
          BARNEY_NYI();
        }
      }
      PRINT(ownGroupVolumes.size());
      if (needRebuild) {
        std::cout << "#bn: running volume _build_ pass" << std::endl;
        if (volumeGeomsGroup) {
          owlGroupRelease(volumeGeomsGroup);
          volumeGeomsGroup = 0;
        }
        volumeGeoms.clear();
        for (auto volume : volumes) 
          volume->build(true);
        volumeGeomsGroup = owlUserGeomGroupCreate
          (owner->devGroup->owl,volumeGeoms.size(),volumeGeoms.data());
        owlGroupBuildAccel(volumeGeomsGroup);
      }
      
      if (needRefit) {
        std::cout << "#bn: running volume _refit_ pass" << std::endl;
        if (volumeGeomsGroup == 0)
          throw std::runtime_error
            ("somebody asked for refit, but there's no group yet!?");
        for (auto volume : volumes) 
          volume->build(false);
        owlGroupRefitAccel(volumeGeomsGroup);
      }

      for (auto volume : ownGroupVolumes) {
        volume->build(true);
      }
    }
  }
  
}
