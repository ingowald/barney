// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

namespace barney {

  /*! the camera model we use in barney */
  struct Camera : public Object {
    typedef std::shared_ptr<Camera> SP;
    struct DD {
      /*! vector from camera center to to lower-left pixel (i.e., pixel
        (0,0)) on the focal plane */
      vec3f dir_00;
      /* vector along u direction, for ONE pixel */
      vec3f dir_du;
      /* vector along v direction, for ONE pixel */
      vec3f dir_dv;
      /*! lens center ... */
      vec3f lens_00;
      /* vector along v direction, for ONE pixel */
      float  lensRadius;
    };
    
    Camera(Context *owner);
    virtual ~Camera() = default;
    
    static Camera::SP create(Context *owner, const char *type);

    DD getDD() { return content; }

    DD content;
  };
    
}

