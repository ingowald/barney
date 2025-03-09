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

namespace BARNEY_NS {

  /*! cylinders with caps, specified through an array of vertices, and
      one array of int2 where each of the two its specified begin and
      end vertex of a cylinder. radii can either come from a separate
      array (if provided), or, i not, use a common radius specified in
      this geometry */
  struct Cones : public Geometry {
    typedef std::shared_ptr<Cones> SP;

    struct DD : public Geometry::DD {
      const vec3f *vertices;
      const vec2i *indices;
      const float *radii;
    };
    
    Cones(Context *context, DevGroup::SP devices);
    virtual ~Cones() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Cones{}"; }
    
    void commit() override;
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool setData(const std::string &member, const Data::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP vertices;
    PODData::SP indices;
    PODData::SP radii;
  };

} // ::BARNEY_NS
