// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"
#include "barney/common/math.h"
#include "barney/Object.h"

namespace BARNEY_NS {

  /*! the camera model we use in barney */
  struct Camera : public barney_api::Camera {
    typedef std::shared_ptr<Camera> SP;
    typedef enum { UNDEFINED=0, PERSPECTIVE, ORTHOGRAPHIC } Type;
    /*! device-data for the camera object; to avoid virtual functions
        this currently uses a 'type'-switch, so the camera code on the
        device will have to 'interpret' what the actual fields
        mean. on the host side, all derived cameras will have to fill
        in this one shared struct */
    struct DD {
      Type  type = UNDEFINED;
      
      /* vector along u direction, for ONE pixel */
      struct {
        /*! vector from camera center to to lower-left pixel (i.e., pixel
          (0,0)) on the focal plane */
        vec3f dir_00;
        vec3f dir_du;
        /* vector along v direction, for ONE pixel */
        vec3f dir_dv;
        /*! lens center ... */
        vec3f lens_00;
        /* radius of lens, for DOF */
        float apertureRadius;
        /* distance to focal plane, for DOF */
        float focusDistance;
      } perspective;
      struct {
        vec3f org_00;
        vec3f org_du;
        vec3f org_dv;
        vec3f dir;
        float height;
        float aspect;
      } orthographic;
    };
    DD dd;

    Camera(Context *owner);
    virtual ~Camera() = default;
    
    static Camera::SP create(Context *owner, const std::string &type);
    
    DD getDD() { return dd; }
  };
    
}

