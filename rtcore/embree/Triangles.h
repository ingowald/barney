#pragma once

#include "rtcore/common/Backend.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/GeomType.h"
#include "rtcore/embree/Geom.h"

namespace barney {
  namespace embree {

    struct TrianglesGeomType;
    
    
    struct TrianglesGeom : public Geom
    {
      TrianglesGeom(TrianglesGeomType *type);

      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(rtc::Buffer *vertices, int numVertices) override;
      void setIndices(rtc::Buffer *indices, int numIndices) override;

      vec3f *vertices = 0;
      int numVertices = 0;
      vec3i *indices = 0;
      int numIndices = 0;
    };
    
  }
}


