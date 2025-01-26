#pragma once

#include "rtcore/common/Backend.h"

namespace barney {
  namespace embree {
    struct TraceInterface;
    struct Device;
    
    typedef void (*AnyHitFct)(embree::TraceInterface &ti);
    typedef void (*ClosestHitFct)(embree::TraceInterface &ti);
    typedef void (*IntersectFct)(embree::TraceInterface &ti);
    typedef void (*BoundsFct)(embree::TraceInterface &ti,
                              const void *gt, box3f &result, int primID);


    struct GeomType : public rtc::GeomType
    {
      GeomType(Device *device,
               const std::string &name,
               size_t sizeOfProgramData,
               bool has_ah,
               bool has_ch);
      AnyHitFct ah = 0;
      ClosestHitFct ch = 0;
      size_t const sizeOfProgramData;
    };
      
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(Device *device,
                        const std::string &name,
                        size_t sizeOfProgramData,
                        bool has_ah,
                        bool has_ch);
      
      virtual ~TrianglesGeomType();
      
      rtc::Geom *createGeom() override;
    };

    struct UserGeomType : public GeomType
    {
      UserGeomType(Device *device,
                   const std::string &name,
                   size_t sizeOfProgramData,
                   bool has_ah,
                   bool has_ch);
      virtual ~UserGeomType();
      
      rtc::Geom *createGeom() override;
      
      BoundsFct     bounds = 0;
      IntersectFct  intersect = 0;
    };
    
  }
}
