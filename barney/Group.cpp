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

namespace barney {
  Group::Group(Context *context, int slot,
               const std::vector<Geometry::SP> &geoms,
               const std::vector<Volume::SP> &volumes)
    : SlottedObject(context,slot),
      geoms(geoms),
      volumes(volumes)
  {}
  
  Group::~Group()
  {
    freeAllGeoms();
  }
  
  /*! implements the parameter set/commit paradigm */
  void Group::commit()
  {}
  
  /*! implements the parameter set/commit paradigm */
  bool Group::setObject(const std::string &member, const Object::SP &value)
  {
    return false;
  }
  
  /*! implements the parameter set/commit paradigm */
  bool Group::setData(const std::string &member, const Data::SP &value)
  {
    if (member == "lights") {
      this->lights = value->as<ObjectRefsData>();
      return true;
    }
    return false;
  }
  
  /*! pretty-printer for printf-debugging */
  std::string Group::toString() const 
  { return "Group"; }

  void Group::freeAllGeoms()
  {
    // we should NOT call owlGeomRelease on these here: they do belong
    // to the Geometry's!
    userGeoms.clear();
    triangleGeoms.clear();
    
    if (triangleGeomGroup) {
      owlGroupRelease(triangleGeomGroup);
      triangleGeomGroup = 0;
    }
    if (userGeomGroup) {
      owlGroupRelease(userGeomGroup);
      userGeomGroup = 0;
    }
  }
  
  void Group::build()
  {
    freeAllGeoms();

    // triangles and user geoms - for now always rebuild
    {
      // userGeoms.clear();
      // triangleGeoms.clear();
      // if (triangleGeomGroup) {
      //   owlGroupRelease(triangleGeomGroup);
      //   triangleGeomGroup = 0;
      // }
      // if (userGeomGroup) {
      //   owlGroupRelease(userGeomGroup);
      //   userGeomGroup = 0;
      // }
      for (auto geom : geoms) {
        if (!geom) continue;
        geom->build();
        for (auto g : geom->triangleGeoms)
          triangleGeoms.push_back(g);
        for (auto g : geom->userGeoms)
          userGeoms.push_back(g);
      }
      if (!userGeoms.empty()) {
        userGeomGroup = owlUserGeomGroupCreate
          (getOWL(),userGeoms.size(),userGeoms.data());
      }
      if (userGeomGroup) {
        owlGroupBuildAccel(userGeomGroup);
      }
      
      if (!triangleGeoms.empty()) {
        triangleGeomGroup = owlTrianglesGeomGroupCreate
          (getOWL(),triangleGeoms.size(),triangleGeoms.data());
      }
      if (triangleGeomGroup) {
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
        if (!volume) continue;
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
      if (needRebuild) {
        // std::cout << "#bn: running volume _build_ pass" << std::endl;
        if (volumeGeomsGroup) {
          owlGroupRelease(volumeGeomsGroup);
          volumeGeomsGroup = 0;
        }
        volumeGeoms.clear();
        for (auto volume : volumes)
          if (volume)
            volume->build(true);
        
        volumeGeomsGroup = owlUserGeomGroupCreate
          (getOWL(),volumeGeoms.size(),volumeGeoms.data());
        owlGroupBuildAccel(volumeGeomsGroup);
      }
      
      if (needRefit) {
        // std::cout << "#bn: running volume _refit_ pass" << std::endl;
        if (volumeGeomsGroup == 0)
          throw std::runtime_error
            ("somebody asked for refit, but there's no group yet!?");
        for (auto volume : volumes)
          if (volume)
            volume->build(false);
        owlGroupRefitAccel(volumeGeomsGroup);
      }

      for (auto volume : ownGroupVolumes) {
        if (volume)
          volume->build(true);
      }
    }
  }
  
}
