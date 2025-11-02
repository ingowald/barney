// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/light/Light.h"

namespace BARNEY_NS {

  struct QuadLight : public Light {
    struct DD {
      vec3f corner{0.f,0.f,0.f};
      vec3f edge0{1.f,0.f,0.f};
      vec3f edge1{0.f,1.f,0.f};
      vec3f emission{1.f,1.f,1.f};
      /*! normal of this lights source; this could obviously be derived
        from cross(edge0,edge1), but is handle to have in a
        renderer */
      vec3f normal;
      /*! area of this lights source; this could obviously be derived
        from cross(edge0,edge1), but is handle to have in a
        renderer */
      float area;
    };

    typedef std::shared_ptr<QuadLight> SP;
    QuadLight(Context *context,
              const DevGroup::SP &devices)
      : Light(context,devices)
    {}

    DD getDD(const affine3f &instanceXfm) const;
    
    std::string toString() const override { return "QuadLight"; }
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    DD staged;
  };
  
}
