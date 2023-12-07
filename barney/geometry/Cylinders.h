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

#include "barney/geometry/Geometry.h"

namespace barney {

  struct DataGroup;

  /*! cylinders with caps, specified through an array of vertices, and
      one array of int2 where each of the two its specified begin and
      end vertex of a cylinder. radii can either come from a separate
      array (if provided), or, i not, use a common radius specified in
      this geometry */
  struct Cylinders : public Geometry {
    typedef std::shared_ptr<Cylinders> SP;

    struct DD {
      const vec3f *points;
      const vec2i *indices;
      const float *radii;
      Material     material;
    };
    
    Cylinders(DataGroup *owner,
              const Material &material,
              const vec3f *points,
              int          numPoints,
              const vec2i *indices,
              int          numIndices,
              const float *radii,
              float        defaultRadius);
    
    static OWLGeomType createGeomType(DevGroup *devGroup);

    OWLBuffer indicesBuffer  = 0;
    OWLBuffer pointsBuffer  = 0;
    OWLBuffer radiiBuffer  = 0;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Cylinders{}"; }
  };

}
