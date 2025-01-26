#pragma once

#include "rtcore/embree/Triangles.h"
#include "embree4/rtcore.h"

namespace barney {
  namespace embree {
  
    struct Group : public rtc::Group {
      Group(Device *device) : rtc::Group(device) 
      {}
      void refitAccel() override { buildAccel(); }
    
      rtc::device::AccelHandle getDD() const override
      {
        rtc::device::AccelHandle dd;
        (const void *&)dd = (const void *)this;
        return dd;
      }
      RTCScene embreeScene = 0;
    };
  
    struct GeomGroup : public Group {
      GeomGroup(Device *device,
                const std::vector<rtc::Geom *> &geoms)
        : Group(device),
          geoms(geoms)
      {}
      
      rtc::Geom *getGeom(int geomID) 
      { assert(geomID >= 0 && geomID < geoms.size()); return geoms[geomID]; }
      
      std::vector<rtc::Geom *> geoms;
    };
      
    struct TrianglesGroup : public GeomGroup {
      TrianglesGroup(Device *device,
                     const std::vector<rtc::Geom *> &geoms);
      
      void buildAccel() override;
      
    };
  
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(Device *device,
                    const std::vector<rtc::Geom *> &geoms)
        : GeomGroup(device,geoms)
      {}

      void buildAccel() override;
    };
  
    struct InstanceGroup : public Group {
      InstanceGroup(Device *device,
                    const std::vector<rtc::Group *> &groups,
                    const std::vector<affine3f>     &xfms);

      GeomGroup *getGroup(int groupID);
      
      void buildAccel() override;
    
      std::vector<rtc::Group*> groups;
      std::vector<affine3f>    xfms;
      std::vector<affine3f>    inverseXfms;
    };
    
  }
    
    
}
