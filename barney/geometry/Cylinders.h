// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/geometry/Geometry.h"

namespace BARNEY_NS {

  /*! cylinders with caps, specified through an array of vertices, and
      one array of int2 where each of the two its specified begin and
      end vertex of a cylinder. radii can either come from a separate
      array (if provided), or, i not, use a common radius specified in
      this geometry */
  struct Cylinders : public Geometry {
    typedef std::shared_ptr<Cylinders> SP;

    struct DD : public Geometry::DD {
      const vec3f *vertices;
      // const vec3f *colors;
      const vec2i *indices;
      const float *radii;
      // int colorPerVertex;
    };
    
    Cylinders(Context *context, DevGroup::SP devices);
    virtual ~Cylinders() = default;
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Cylinders{}"; }
    
    void commit() override;
    
    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    bool set1i(const std::string &member, const int &value) override;
    bool set1f(const std::string &member, const float &value) override;
    bool setData(const std::string &member, const barney_api::Data::SP &value) override;
    bool setObject(const std::string &member, const Object::SP &value) override;
    /*! @} */
    // ------------------------------------------------------------------

    PODData::SP vertices;
    PODData::SP indices;
    PODData::SP radii;
  };

}
