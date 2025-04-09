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

#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Group.h"
#include "rtcore/cuda/Geom.h"
#include "rtcore/cuda/GeomType.h"
#include "rtcore/cuda/Buffer.h"

namespace rtc {
  namespace cuda {
    
    rtc::AccelHandle getAccelHandle(Group *ig)
    { return ig->getDD(); }
    
    Device::~Device()
    {}
    
    void Device::freeGroup(Group *g)
    { delete g; }

    Denoiser *Device::createDenoiser()
    { return nullptr; }

    Buffer *Device::createBuffer(size_t numBytes,
                                 const void *initValues)
    {
      return new Buffer(this,numBytes,initValues);
    }

    void Device::freeBuffer(Buffer *b)
    {
      delete b;
    }

    void Device::freeGeomType(GeomType *gt)
    {
      delete gt;
    }

    void Device::freeGeom(Geom *g)
    {
      delete g;
    }
      
    Group *
    Device::createTrianglesGroup(const std::vector<Geom *> &geoms)
    {
      return new TrianglesGeomGroup(this,geoms);
    }
      
    Group *
    Device::createUserGeomsGroup(const std::vector<Geom *> &geoms)
    {
      return new UserGeomGroup(this,geoms);
    }
    
    Group *
    Device::createInstanceGroup(const std::vector<Group *> &groups,
                                const std::vector<int>      &instIDs,
                                const std::vector<affine3f> &xfms)
    {
      return new InstanceGroup(this,groups,instIDs,xfms);
    }

    void Device::buildPipeline()
    {
    }
    
    void Device::buildSBT()
    {
    }

  }    
}
