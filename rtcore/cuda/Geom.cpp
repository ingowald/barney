// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Geom.h"
#include "rtcore/cuda/GeomType.h"

namespace rtc {
  namespace cuda {

    Geom::Geom(GeomType *gt)
      : gt(gt),
        data(gt->sizeOfDD)
    {}
    
    Geom::~Geom()
    {}
    
    void Geom::setDD(const void *dd)
    {
      memcpy(data.data(),dd,data.size());
    }



    TrianglesGeom::TrianglesGeom(GeomType *gt)
      : Geom(gt)
    {}
      
    void TrianglesGeom::setVertices(Buffer *vertices,
                                    int numVertices)
    {
      this->vertices = vertices;
      this->numVertices = numVertices;
    }
      
    void TrianglesGeom::setIndices(Buffer *indices,
                                   int numIndices)
    {
      this->indices = indices;
      this->numIndices = numIndices;
    }


    UserGeom::UserGeom(GeomType *gt)
      : Geom(gt)
    {}
      
    void UserGeom::setPrimCount(int primCount)
    {
      this->primCount = primCount;
    }
    
  }
}
