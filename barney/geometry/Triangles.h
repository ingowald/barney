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

  /*! geometry in the form of a regular triangle mesh - vertex
      positoins array, vertex indices array, verex normals, and
      texcoords */
  struct Triangles : public Geometry {
    typedef std::shared_ptr<Triangles> SP;

    struct DD {
      int          numIndices;
      int          numVertices;
      const vec3i *indices;
      const vec3f *vertices;
      const vec3f *normals;
      const vec2f *texcoords;
      Material     material;
    };
    
    Triangles(DataGroup *owner,
              const Material &material,
              int numIndices,
              const vec3i *indices,
              int numVertices,
              const vec3f *vertices,
              const vec3f *normals,
              const vec2f *texcoords);
    void update(const Material &material,
                int numIndices,
                const vec3i *indices,
                int numVertices,
                const vec3f *vertices,
                const vec3f *normals,
                const vec2f *texcoords);
    
    static OWLGeomType createGeomType(DevGroup *devGroup);

    int       numIndices;
    int       numVertices;
    OWLBuffer indicesBuffer  = 0;
    OWLBuffer verticesBuffer = 0;

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Triangles{}"; }
  };

}