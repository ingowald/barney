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
#include "barney/Context.h"

namespace BARNEY_NS {

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
    : barney_api::Group(context),
      devices(devices),
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
        rtc->freeGroup(pld->triangleGeomGroup);
        pld->triangleGeomGroup = 0;
      }
      if (pld->userGeomGroup) {
        rtc->freeGroup(pld->userGeomGroup);
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
      // bool needRefit = false;
      // bool needRebuild = false;
      
      // ------------------------------------------------------------------
      // clear all pld data
      // ------------------------------------------------------------------
      for (auto device : *devices) {
        auto myPLD = getPLD(device);
        auto rtc = device->rtc;
        // if we have a volume geoms _group_, then it is _us_ that
        // created this, and we have to free it.
        if (myPLD->volumeGeomsGroup) {
          rtc->freeGroup(myPLD->volumeGeomsGroup);
          myPLD->volumeGeomsGroup = 0;
        }

        // the volume _geoms_ are created/owned by the actual volume
        // accel, and it is _their_ job to free those if required.
        myPLD->volumeGeoms.clear();
        
        // the volume _groups_ are created/owned by the actual volume
        // accel, and it is _their_ job to free those if required.
        myPLD->volumeGroups.clear();
      }
      
      // ------------------------------------------------------------------
      // rebuild the volumes themselves. the result is, with each
      // volume, two (possibly) empty sets of generated geoms and
      // volumes, respectively, that we can now gather here.
      // ------------------------------------------------------------------
      for (auto volume : volumes)
        if (volume)
          volume->build(true);

      // ------------------------------------------------------------------
      // now that all volumes (and their accels) have been built, go
      // over all those and gather the generated volume geoms and
      // volume grousp
      // ------------------------------------------------------------------
      for (auto device : *devices) {
        PLD *myPLD = getPLD(device);
        for (auto volume : volumes) {
          Volume::PLD *volumePLD = volume->getPLD(device);
          // gather all geoms from this group (if any)
          for (auto geom : volumePLD->generatedGeoms)
            myPLD->volumeGeoms.push_back(geom);
          // gather all geoms from this group (if any)
          for (auto group : volumePLD->generatedGroups) {
            assert(group);
            myPLD->volumeGroups.push_back(group);
          }
        }

        // now we have all rtc geoms and all rtc groups across all the
        // volumes that have been in our current group.
        if (!myPLD->volumeGeoms.empty())
          printf("todo: build volume geoms group, and add it to root\n");
      }
    }
  }
  
}
