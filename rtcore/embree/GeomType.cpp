#include "rtcore/embree/GeomType.h"
#include "rtcore/embree/Triangles.h"


namespace barney {
  namespace embree {

    GeomType::GeomType(Device *device,
                       const std::string &name,
                       size_t sizeOfProgramData,
                       bool has_ah,
                       bool has_ch)
      : rtc::GeomType(device),
        sizeOfProgramData(sizeOfProgramData)
    {
      if (has_ah)
        ah = (AnyHitFct)rtc::getSymbol("barney_embree_ah_"+name);
      if (has_ch)
        ch = (ClosestHitFct)rtc::getSymbol("barney_embree_ch_"+name);
    }
    
    TrianglesGeomType::TrianglesGeomType(Device *device,
                                         const std::string &name,
                                         size_t sizeOfProgramData,
                                         bool has_ah,
                                         bool has_ch)
      : GeomType(device,name,sizeOfProgramData,has_ah,has_ch)
    {
    }

    UserGeomType::UserGeomType(Device *device,
                                         const std::string &name,
                                         size_t sizeOfProgramData,
                                         bool has_ah,
                                         bool has_ch)
      : GeomType(device,name,sizeOfProgramData,has_ah,has_ch)
    {
      intersect = (IntersectFct)rtc::getSymbol("barney_embree_intersect_"+name);
      bounds    = (BoundsFct)rtc::getSymbol("barney_embree_bounds_"+name);
    }

    
    UserGeomType::~UserGeomType()
    {}
    
    TrianglesGeomType::~TrianglesGeomType()
    {}
    
    rtc::Geom *TrianglesGeomType::createGeom()
    { return new TrianglesGeom(this); }

    rtc::Geom *UserGeomType::createGeom()
    { BARNEY_NYI(); }
    
  }
}
