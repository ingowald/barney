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

namespace barney {

  /*! A geometry made of multiple "capsules", where each capsule is
      "pill-like" shape obtained by linearly connecting two
      spheres. Unlike cylinders both end-points of the capsule have
      their own radius that the rest of the shape linearly
      interpolates between; capsules also always have "rounded caps"
      in the sense that both of the end points form complete
      spheres. Capsules can also interpolate a "color" attribute.

      Is defined by three parameters:

      `int2 radii[]` two vertex indices per prim, specifying end point
      position and radii for each capsule.

      `float3 vertices[]` position (.xyz) and radius (.w) of each vertex
  */
  struct Capsules : public Geometry {
    typedef std::shared_ptr<Capsules> SP;

    struct DD : public Geometry::DD {
      const vec4f *vertices;
      const vec2i *indices;
    };
    
    Capsules(Context *context, int slot);
    virtual ~Capsules() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Capsules{}"; }
    
    void commit() override;
    
    static rtc::GeomType *createGeomType(DevGroup *devGroup);

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool setData(const std::string &member, const Data::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP vertices;
    PODData::SP indices;
  };

}
