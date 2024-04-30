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

#pragma once

#include "barney/geometry/Geometry.h"
#include "barney/volume/Volume.h"
// #include "barney/MultiPass.h"
#include "barney/common/Data.h"

namespace barney {

  /*! a logical "group" of objects in a data group -- i.e., geometries
      and volumes (and maybe, eventual, lights?) -- that can be
      instantiated */
  struct Group : public Object {
    typedef std::shared_ptr<Group> SP;

    Group(ModelSlot *owner,
          const std::vector<Geometry::SP> &geoms,
          const std::vector<Volume::SP> &volumes);
    virtual ~Group();

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    bool setData(const std::string &member, const Data::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    void build();

    void freeAllGeoms();
    void freeAllVolumes();
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override;

    ModelSlot *const owner;
    const std::vector<Volume::SP>   volumes;
    const std::vector<Geometry::SP> geoms;

    /*! lights assigned to this group */
    ObjectRefsData::SP lights;
    
    std::vector<OWLGeom> triangleGeoms;
    std::vector<OWLGeom> userGeoms;
    std::vector<OWLGeom> volumeGeoms;
    OWLGroup userGeomGroup     = 0;
    OWLGroup triangleGeomGroup = 0;
    OWLGroup volumeGeomsGroup  = 0;
    
    // std::vector<MultiPass::Object::SP> multiPassObjects;
  };
  
}
