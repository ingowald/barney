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

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace cuda {

    struct Device;
    
    struct Group {
      Group(Device *device);
      virtual ~Group();
      virtual void buildAccel() = 0;
      virtual void refitAccel() { buildAccel(); }
      rtc::device::AccelHandle getDD() const { return (rtc::device::AccelHandle)d_accel; }

      Device *const device;
      void *d_accel = 0;
    };

    struct InstanceGroup : public Group {
      InstanceGroup(Device *device,
                    const std::vector<Group *> &groups,
                    const std::vector<affine3f> &xfms);
      void buildAccel() override;
    };
    
    struct GeomGroup : public Group {
      GeomGroup(Device *device);
    };

    struct TrianglesGeomGroup : public GeomGroup {
      TrianglesGeomGroup(Device *device, const std::vector<Geom *> &geoms);
      void buildAccel() override;
    };
    
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(Device *device, const std::vector<Geom *> &geoms);
      void buildAccel() override;
    };
    
  }
}
