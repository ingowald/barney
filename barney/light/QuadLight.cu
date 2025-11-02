// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "QuadLight.h"

namespace BARNEY_NS {
  
  QuadLight::DD QuadLight::getDD(const affine3f &instanceXfm) const
  {
    DD dd;
    dd.corner = xfmPoint(instanceXfm,staged.corner);
    dd.edge0 = xfmVector(instanceXfm,staged.edge0);
    dd.edge1 = xfmVector(instanceXfm,staged.edge1);
    dd.emission = staged.emission;
    dd.area = length(cross(dd.edge0,dd.edge1));
    return dd;
  }
  
  bool QuadLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "corner") {
      staged.corner = value;
      return true;
    }
    if (member == "edge0") {
      staged.edge0 = value;
      return true;
    }
    if (member == "edge1") {
      staged.edge1 = value;
      return true;
    }
    if (member == "emission") {
      staged.emission = value;
      return true;
    }
    return false;
  }

}
