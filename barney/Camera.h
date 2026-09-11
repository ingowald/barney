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
    typedef enum {
      UNDEFINED=0,
      PERSPECTIVE,
      ORTHOGRAPHIC,
      OMNIDIRECTIONAL
    } Type;
    /*! device-data for the camera object; to avoid virtual functions
        this currently uses a 'type'-switch, so the camera code on the
        device will have to 'interpret' what the actual fields
        mean. on the host side, all derived cameras will have to fill
        in this one shared struct */
    struct DD {
      inline DD() : type(UNDEFINED) {};
      inline DD(const DD &) = default;
      inline ~DD() = default;

      Type  type = UNDEFINED;

      /* Motion-vector reprojection matrices (world -> clip). When
         haveMotionMatrices is false, BN_FB_MOTION writes are skipped
         regardless of whether per-instance deltas were supplied. */
      bool     haveMotionMatrices = false;
      float    currViewProj[16]   = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      float    prevViewProj[16]   = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

      union {
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
        struct {
          affine3f toWorld;
        } omni;
      };
    };
    DD dd;

    Camera(Context *owner);
    virtual ~Camera() = default;

    static Camera::SP create(Context *owner, const std::string &type);

    /*! subtypes call from their commit() so motion params propagate
        into DD after subtype-specific fields are populated */
    void commitMotionFields();

    bool set1i(const std::string &member, const int &value) override;
    bool set4x4f(const std::string &member, const vec4f *value) override;

    DD getDD() { return dd; }

  protected:
    bool  motionEnabled = false;
    float motionCurrViewProj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float motionPrevViewProj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  };
    
}

