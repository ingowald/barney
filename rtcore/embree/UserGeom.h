#pragma once

#include "rtcore/common/Backend.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/Geom.h"

namespace barney {
  namespace embree {

    struct UserGeom : public Geom
    {
      UserGeom(UserGeomType *type) : Geom(type) {};
      
      /*! only for user geoms */
      void setPrimCount(int primCount) override{};
      /*! can only get called on triangle type geoms */
      void setVertices(rtc::Buffer *vertices, int numVertices) override {};
      void setIndices(rtc::Buffer *indices, int numIndices) override{};

      int primCount = 0;
    };
    
  }
}


