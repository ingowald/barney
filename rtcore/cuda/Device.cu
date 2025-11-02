// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


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
