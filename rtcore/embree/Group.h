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

#include "rtcore/embree/Triangles.h"
#include "embree4/rtcore.h"

namespace rtc {
  namespace embree {
  
    struct Group {
      Group(Device *device)
        : device(device)
      {}
      void refitAccel() { buildAccel(); }
      virtual void buildAccel() = 0;
      
      rtc::device::AccelHandle getDD() const
      {
        rtc::device::AccelHandle dd;
        (const void *&)dd = (const void *)this;
        return dd;
      }
      RTCScene embreeScene = 0;
      Device *const device;
    };
    
    struct GeomGroup : public Group {
      GeomGroup(Device *device,
                const std::vector<Geom *> &geoms)
        : Group(device),
          geoms(geoms)
      {}
      
      Geom *getGeom(int geomID) 
      { assert(geomID >= 0 && geomID < geoms.size()); return geoms[geomID]; }
      
      std::vector<Geom *> geoms;
    };
      
    struct TrianglesGroup : public GeomGroup {
      TrianglesGroup(Device *device,
                     const std::vector<Geom *> &geoms);
      
      void buildAccel() override;
      
    };
  
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(Device *device,
                    const std::vector<Geom *> &geoms)
        : GeomGroup(device,geoms)
      {}

      void buildAccel() override;
    };
  
    struct InstanceGroup : public Group {
      InstanceGroup(Device *device,
                    const std::vector<Group *> &groups,
                    const std::vector<affine3f>     &xfms);

      GeomGroup *getGroup(int groupID);
      
      void buildAccel() override;
    
      std::vector<Group*> groups;
      std::vector<affine3f>    xfms;
      std::vector<affine3f>    inverseXfms;
    };
    
  }
    
    
}
