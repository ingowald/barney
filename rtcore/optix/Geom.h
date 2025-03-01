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

#pragma once

#include "rtcore/optix/Device.h"
#include "rtcore/optix/Buffer.h"
#include <owl/owl.h>

namespace rtc {
  namespace optix {
    struct Device;
    
    struct GeomType;
    
    struct Geom {
      Geom(GeomType *gt, OWLGeom geom);
      virtual ~Geom();
      void setDD(const void *dd);

      virtual void setPrimCount(int primCount) = 0;
      virtual void setVertices(Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(Buffer *indices, int numIndices) = 0;
      
      GeomType *const gt;
      OWLGeom   const owl;
    };

    struct TrianglesGeom : public optix::Geom {
      TrianglesGeom(GeomType *gt, OWLGeom geom);
      
      void setPrimCount(int primCount) override { assert(0); }
      void setVertices(Buffer *vertices, int numVertices) override;
      void setIndices(Buffer *indices, int numIndices) override;
      /*! only for user geoms */
      /*! can only get called on triangle type geoms */
    };
    struct UserGeom : public optix::Geom {
      UserGeom(GeomType *gt, OWLGeom geom);
      
      void setPrimCount(int primCount);
      void setVertices(Buffer *vertices, int numVertices) override
      { assert(0); }
      void setIndices(Buffer *indices, int numIndices) override
      { assert(0); }
    };
    
    struct GeomType {
      GeomType(optix::Device *device);
      virtual ~GeomType();
      
      virtual Geom *createGeom() = 0;
      
      OWLGeomType gt = 0;
      optix::Device *const device;
    };
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(optix::Device *device,
                        const std::string &ptxCode,
                        const std::string &typeName,
                        size_t sizeOfDD,
                        bool has_ah, bool has_ch);
      Geom *createGeom() override;
    };
    struct UserGeomType : public GeomType
    {
      UserGeomType(optix::Device *device,
                   const std::string &ptxCode,
                   const std::string &typeName,
                   size_t sizeOfDD,
                   bool has_ah, bool has_ch);
      Geom *createGeom() override;
    };
    
  }
}
