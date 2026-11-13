// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/geometry/Geometry.h"
#include "native/volume/Volume.h"
// #include "native/MultiPass.h"
#include "native/common/Data.h"

namespace BARNEY_NS {
  namespace native {

    /*! a logical "group" of objects in a data group -- i.e., geometries
      and volumes (and maybe, eventual, lights?) -- that can be
      instantiated */
    struct Group : public Object {
      typedef std::shared_ptr<Group> SP;

      Group(Device *device,
            // Context *context,
            // const DevGroup::SP &devices,
            const std::vector<std::shared_ptr<Geometry>> &geoms,
            const std::vector<std::shared_ptr<Volume>> &volumes);
      ~Group() override;

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      void commit() override;
      bool setObject(const std::string &member, const Object::SP &value) override;
      bool setData(const std::string &member,
                   const Data::SP &value) override;
      /*! @} */
      // ------------------------------------------------------------------
    
      virtual void build();

      void freeAllGeoms();
      void freeAllVolumes();
    
      /*! pretty-printer for printf-debugging */
      std::string toString() const override;

      const std::vector<Volume::SP>   volumes;
      const std::vector<Geometry::SP> geoms;

      /*! lights assigned to this group */
      ObjectRefsData::SP lights;

// #if 1
//       struct /* per logical device */PLD {
      struct {
        std::vector<rtc::Geom *> triangleGeoms;
        std::vector<rtc::Geom *> userGeoms;
        std::vector<rtc::Geom *> volumeGeoms;
        std::vector<rtc::Group *> volumeGroups;
        rtc::Group *userGeomGroup     = 0;
        rtc::Group *triangleGeomGroup = 0;
        rtc::Group *volumeGeomsGroup  = 0;
      } rtc;
//       };
//       PLD *getPLD(Device *device);
    
//       std::vector<PLD> perLogical;
// #else
//       std::vector<OWLGeom> triangleGeoms;
//       std::vector<OWLGeom> userGeoms;
//       std::vector<OWLGeom> volumeGeoms;
//       OWLGroup userGeomGroup     = 0;
//       OWLGroup triangleGeomGroup = 0;
//       OWLGroup volumeGeomsGroup  = 0;
// #endif
      // DevGroup::SP const devices;
      // std::vector<MultiPass::Object::SP> multiPassObjects;
    };

  }
}
