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

#pragma once

#include "barney/Geometry.h"

namespace barney {

  struct Group : public Object {
    typedef std::shared_ptr<Group> SP;

    Group(DataGroup *owner,
          const std::vector<Geometry::SP> &geoms);
    
    void build();

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Group{}"; }

    OWLGroup userGeomGroup = 0;
    OWLGroup triangleGeomGroup = 0;
    std::vector<OWLGroup> secondPassGroups;
    DataGroup *const owner;
    const std::vector<Geometry::SP> geoms;
  };
  
}