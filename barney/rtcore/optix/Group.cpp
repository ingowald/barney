// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/optix/Group.h"
#include "rtcore/optix/Device.h"

namespace BARNEY_NS {
  namespace rtc {

    Group::Group(Device *device, OWLGroup owl)
      : device(device),
        owl(owl)
    {}
    
    rtc::AccelHandle Group::getDD() const
    {
      OptixTraversableHandle handle
        = owlGroupGetTraversable(owl,0);
      return (const rtc::AccelHandle &)handle;
    }
    
    void Group::buildAccel()
    {
      owlGroupBuildAccel(owl);
    }
    
    void Group::refitAccel()
    {
      owlGroupRefitAccel(owl);
    }

    void Group::setTransforms(const std::vector<affine3f> &xfms)
    {
      owlInstanceGroupSetTransforms(owl, 0,
                                    (const float *)xfms.data(),
                                    OWL_MATRIX_FORMAT_OWL);
    }

  }
}
