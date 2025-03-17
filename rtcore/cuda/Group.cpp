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

#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Group.h"

namespace rtc {
  namespace cuda {

    Group::Group(Device *device)
      : device(device)
    {}
    
    Group::~Group()
    {}

    InstanceGroup::InstanceGroup(Device *device,
                                 const std::vector<Group *> &groups,
                                 const std::vector<affine3f> &xfms)
      : Group(device)
    {
      PING; throw std::runtime_error("not implemented");
    }

    void InstanceGroup::buildAccel() 
    {
      PING; throw std::runtime_error("not implemented");
    }
    
    GeomGroup::GeomGroup(Device *device)
      : Group(device)
    {}

    TrianglesGeomGroup::TrianglesGeomGroup(Device *device,
                                           const std::vector<Geom *> &geoms)
      : GeomGroup(device)
    {
      PING; throw std::runtime_error("not implemented");
    }

    void TrianglesGeomGroup::buildAccel() 
    {
      PING; throw std::runtime_error("not implemented");
    }
    
    UserGeomGroup::UserGeomGroup(Device *device,
                                 const std::vector<Geom *> &geoms)
      : GeomGroup(device)
    {
      PING; throw std::runtime_error("not implemented");
    }

    void UserGeomGroup::buildAccel() 
    {
      PING; throw std::runtime_error("not implemented");
    }

    
  }
}
