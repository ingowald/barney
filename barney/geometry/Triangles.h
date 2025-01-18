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

  struct ModelSlot;


  // ==================================================================
  /*! Scalar field made of 3D structured data, constting of Nx*Ny*Nz
      scalars.

      Supported settable fields:

      - "vertices"  (BNData<float3>)
      - "indices"   (BNData<int3>)
      - "normals"   (BNData<float3>)
      - "texcoords" (BNData<float2>)
  */
  struct Triangles : public Geometry {
    typedef std::shared_ptr<Triangles> SP;

    struct DD : public Geometry::DD {
      const vec3i *indices;
      const vec3f *vertices;
      const vec3f *normals;
      const vec2f *texcoords;
      // const vec4f *vertexAttribute[5];
    };
    
    Triangles(Context *context, int slot);
    virtual ~Triangles();
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Triangles{}"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool setData(const std::string &member, const Data::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    static rtc::GeomType *createGeomType(DevGroup *devGroup);

    PODData::SP vertices;
    PODData::SP indices;
    PODData::SP normals;
    // TODO: do we still need this in times of ANARI?
    PODData::SP texcoords;
  };

}
