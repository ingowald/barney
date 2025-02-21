// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/common/barney-common.h"
#include "barney/Object.h"

namespace BARNEY_NS {

  /*! the camera model we use in barney */
  struct Camera : public barney_api::ParameterizedObject {
    typedef std::shared_ptr<Camera> SP;
    typedef enum { UNDEFINED=0, PERSPECTIVE } Type;
    /*! device-data for the camera object; to avoid virtual functions
        this currently uses a 'type'-switch, so the camera code on the
        device will have to 'interpret' what the actual fields
        mean. on the host side, all derived cameras will have to fill
        in this one shared struct */
    struct DD {
      Type  type = UNDEFINED;
      
      /*! vector from camera center to to lower-left pixel (i.e., pixel
        (0,0)) on the focal plane */
      vec3f dir_00;
      /* vector along u direction, for ONE pixel */
      vec3f dir_du;
      /* vector along v direction, for ONE pixel */
      vec3f dir_dv;
      /*! lens center ... */
      vec3f lens_00;
      /* radius of lens, for DOF */
      float lensRadius;
      /* distance to focal plane, for DOF */
      float focalLength;
    };
    DD dd;

    struct {
      float focalLength = 0.f;
      float lensRadius = 0.f;
    } defaultValues;
    
    Camera(Context *owner);
    virtual ~Camera() = default;
    
    static Camera::SP create(Context *owner, const std::string &type);
    
    DD getDD() { return dd; }
  };
    
}

