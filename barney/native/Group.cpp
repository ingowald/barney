// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/ModelSlot.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {

    // Group::PLD *Group::getPLD(Device *device)
    // {
    //   assert(device);
    //   assert(device->contextRank() >= 0);
    //   assert(device->contextRank() < perLogical.size());
    //   return &perLogical[device->contextRank()];
    // }


    Group::Group(Device *device,
                 const std::vector<Geometry::SP> &geoms,
                 const std::vector<Volume::SP> &volumes)
      : Object(device),
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
      if (rtc.triangleGeomGroup) {
        device->rtc->freeGroup(rtc.triangleGeomGroup);
        rtc.triangleGeomGroup = 0;
      }
      if (rtc.userGeomGroup) {
        device->rtc->freeGroup(rtc.userGeomGroup);
        rtc.userGeomGroup = 0;
      }
      if (rtc.volumeGeomsGroup) {
        device->rtc->freeGroup(rtc.volumeGeomsGroup);
        rtc.volumeGeomsGroup = 0;
      }
      rtc.userGeoms.clear();
      rtc.triangleGeoms.clear();
      rtc.volumeGeoms.clear();
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
        for (auto geom : geoms) {
          for (auto g : geom->rtc.triangleGeoms)
            rtc.triangleGeoms.push_back(g);
          for (auto g : geom->rtc.userGeoms)
            rtc.userGeoms.push_back(g);
        }
        
        if (!rtc.userGeoms.empty()) {
          rtc.userGeomGroup
            = device->rtc->createUserGeomsGroup(rtc.userGeoms);
          rtc.userGeomGroup->buildAccel();
        }
        
        if (!rtc.triangleGeoms.empty()) {
          rtc.triangleGeomGroup
            = device->rtc->createTrianglesGroup(rtc.triangleGeoms);
          rtc.triangleGeomGroup->buildAccel();
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
        // if we have a volume geoms _group_, then it is _us_ that
        // created this, and we have to free it.
        if (rtc.volumeGeomsGroup) {
          device->rtc->freeGroup(rtc.volumeGeomsGroup);
          rtc.volumeGeomsGroup = 0;
        }

        // the volume _geoms_ are created/owned by the actual volume
        // accel, and it is _their_ job to free those if required.
        rtc.volumeGeoms.clear();
        
        // the volume _groups_ are created/owned by the actual volume
        // accel, and it is _their_ job to free those if required.
        rtc.volumeGroups.clear();
        
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
        for (auto volume : volumes) {
          // // gather all geoms from this group (if any)
          // for (auto geom : volume->generatedGeoms)
          //   rtc.volumeGeoms.push_back(geom);
          // gather all geoms from this group (if any)
          for (auto group : volume->generatedGroups) 
            rtc.volumeGroups.push_back(group);
        }
      }
    }

  }
}
