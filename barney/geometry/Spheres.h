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

  struct Spheres : public Geometry {
    typedef std::shared_ptr<Spheres> SP;

    struct DD : public Geometry::DD {
      vec3f       *origins;
      float       *radii;
      vec3f       *colors;
      float        defaultRadius;
      // const vec4f *vertexAttribute[5];
    };

    Spheres(SlotContext *slotContext);
    
    static rtc::GeomType *createGeomType(rtc::Device *device,
                                         const void *);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Spheres{}"; }

    void commit() override;
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1f(const std::string &member, const float &value) override;
    bool setData(const std::string &member, const barney_api::Data::SP &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP origins = 0;
    PODData::SP colors  = 0;
    PODData::SP radii   = 0;
    float       defaultRadius = .1f;
  };
  
}
  
