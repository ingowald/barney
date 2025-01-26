#include "rtcore/embree/Triangles.h"

namespace barney {
  namespace embree {

    Geom::Geom(GeomType *type)
      : rtc::Geom(type->device),
        type(type),
        programData(type->sizeOfProgramData)
    {}
    
    TrianglesGeom::TrianglesGeom(TrianglesGeomType *type)
      : Geom(type)
    {}
    
    /*! only for user geoms */
    void TrianglesGeom::setPrimCount(int primCount)
    {
      throw std::runtime_error("setPrimCount only makes sense for user geoms");
    }
    
    /*! can only get called on triangle type geoms */
    void TrianglesGeom::setVertices(rtc::Buffer *vertices,
                                    int numVertices)
    {
      this->vertices = (vec3f*)((Buffer *)vertices)->mem;
      this->numVertices = numVertices;
    }
    
    void TrianglesGeom::setIndices(rtc::Buffer *indices, int numIndices)
    {
      this->indices = (vec3i*)((Buffer *)indices)->mem;
      this->numIndices = numIndices;
    }
    
    void Geom::setDD(const void *dd)
    {
      memcpy(programData.data(),dd,programData.size());
    }

  }
}
