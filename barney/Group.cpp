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

  Group::PLD *Group::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }


  Group::Group(Context *context, 
               const DevGroup::SP &devices, 
               const std::vector<Geometry::SP> &geoms,
               const std::vector<Volume::SP> &volumes)
    : SlottedObject(context,devices),
      geoms(geoms),
      volumes(volumes)
  {
    perLogical.resize(devices->numLogical);
  }
  
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
    // userGeoms.clear();
    // triangleGeoms.clear();

    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      if (pld->triangleGeomGroup) {
        rtc->free(pld->triangleGeomGroup);
        pld->triangleGeomGroup = 0;
      }
      if (pld->userGeomGroup) {
        rtc->free(pld->userGeomGroup);
        pld->userGeomGroup = 0;
      }
    }
  }
  
  void Group::build()
  {
    freeAllGeoms();

    // ==================================================================
    // triangles and user geoms - for now always rebuild
    // ==================================================================
    {
      // first, let them all build/update themselves
      for (auto geom : geoms) {
        if (!geom) continue;
        geom->build();
      }
      
      // now, do our stuff on a per-device basis
      for (auto device : *devices) {
        PLD *myPLD = getPLD(device);
        for (auto geom : geoms) {
          Geometry::PLD *geomPLD = geom->getPLD(device);
          for (auto g : geomPLD->triangleGeoms)
            myPLD->triangleGeoms.push_back(g);
          for (auto g : geomPLD->userGeoms)
            myPLD->userGeoms.push_back(g);
        }
        
        if (!myPLD->userGeoms.empty()) {
          myPLD->userGeomGroup
            = device->rtc->createUserGeomsGroup(myPLD->userGeoms);
          myPLD->userGeomGroup->buildAccel();
        }
        
        if (!myPLD->triangleGeoms.empty()) {
          myPLD->triangleGeomGroup
            = device->rtc->createTrianglesGroup(myPLD->triangleGeoms);
          myPLD->triangleGeomGroup->buildAccel();
        }
      }
    }
    // ==================================================================
    // volumes - these may need two passes
    // ==================================================================
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
        // ------------------------------------------------------------------
        // clear all pld data
        // ------------------------------------------------------------------
        for (auto device : *devices) {
          auto myPLD = getPLD(device);
          auto rtc = device->rtc;
          if (myPLD->volumeGeomsGroup) {
            // owlGroupRelease(volumeGeomsGroup);
            rtc->free(myPLD->volumeGeomsGroup);
            myPLD->volumeGeomsGroup = 0;
          }
          myPLD->volumeGeoms.clear();
        }
        // ------------------------------------------------------------------
        // rebuidl the volumes themselves
        // ------------------------------------------------------------------
        for (auto volume : volumes)
          if (volume)
            volume->build(true);
        // ------------------------------------------------------------------
        // and rebuild our pld data
        // ------------------------------------------------------------------
        for (auto device : *devices) {
          PLD *myPLD = getPLD(device);
          auto rtc = device->rtc;
          for (auto volume : volumes)
            Volume::PLD *volumePLD = volume->getPLD(device);
          myPLD->volumeGeomsGroup
            = rtc->createUserGeomsGroup(myPLD->volumeGeoms);
          myPLD->volumeGeomsGroup->buildAccel();
        }
      }
      
      if (needRefit) {
        // std::cout << "#bn: running volume _refit_ pass" << std::endl;
        for (auto volume : volumes)
          volume->build(false);
        // owlGroupRefitAccel(volumeGeomsGroup);
        for (auto device : *devices) 
          getPLD(device)->volumeGeomsGroup->refitAccel();
      }
      
      for (auto volume : ownGroupVolumes) 
        for (auto device : *devices) 
          volume->build(true);
    }
  }
  
}
