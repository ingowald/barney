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

#include "barney/Camera.h"
#include "barney/Context.h"

namespace BARNEY_NS {

  Camera::Camera(Context *owner)
    : barney_api::Camera(owner)
  {}

  // ##################################################################

  struct PerspectiveCamera : public Camera {
    PerspectiveCamera(Context *owner)
      : Camera(owner)
    {}
    virtual ~PerspectiveCamera() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    vec3f position  { 0, 0, 0 };
    vec3f direction { 0, 0, 1 };
    vec3f up        { 0, 1, 0 };
    float aspect    = 1.f;
    float fovy      = 60.f;
    float focusDistance  = 0.f;
    float apertureRadius = 0.f;
  };
  
  bool PerspectiveCamera::set1f(const std::string &member, const float &value)
  {
    if (Camera::set1f(member,value))
      return true;
    if (member == "aspect") {
      aspect = value;
      return true;
    }
    if (member == "apertureRadius") {
      apertureRadius = value;
      return true;
    }
    if (member == "focusDistance") {
      focusDistance = value;
      return true;
    }
    if (member == "fovy") {
      fovy = value;
      return true;
    }
    return false;
  }
  
  bool PerspectiveCamera::set3f(const std::string &member, const vec3f &value)
  {
    if (Camera::set3f(member,value))
      return true;
    if (member == "position") {
      position = value;
      return true;
    }
    if (member == "direction") {
      direction = normalize(value);
      return true;
    }
    if (member == "up") {
      up = value;
      return true;
    }
    return false;
  }
    
  void PerspectiveCamera::commit()
  {
    vec3f from   = (const vec3f&)position;
    vec3f dir_00 = normalize(direction);
    
    vec3f dir_du = normalize(cross(dir_00, up));
    vec3f dir_dv = normalize(cross(dir_du, dir_00));

    dir_00 *= (float)(1.f / (2.0f * tanf((0.5f * fovy) * (float)M_PI / 180.0f)));
    // dir_00 -= 0.5f * dir_du;
    // dir_00 -= 0.5f * dir_dv;

    dd.type = Camera::PERSPECTIVE;
    dd.perspective.dir_00  = dir_00;
    dd.perspective.dir_du  = dir_du;
    dd.perspective.dir_dv  = dir_dv;
    dd.perspective.lens_00 = from;
    dd.perspective.focusDistance = focusDistance;
    dd.perspective.apertureRadius = apertureRadius;
  }
    

  // ##################################################################

  struct OrthographicCamera : public Camera {
    OrthographicCamera(Context *owner)
      : Camera(owner)
    {}
    virtual ~OrthographicCamera() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    bool set1f(const std::string &member, const float &value) override;
    bool set3f(const std::string &member, const vec3f &value) override;
    /*! @} */
    // ------------------------------------------------------------------
    
    vec3f position  { 0, 0, 0 };
    vec3f direction { 0, 0, 1 };
    vec3f up        { 0, 1, 0 };
    float aspect    = 1.f;
    float height    = 1.f;
    float near      = 0.f;
    float far       = BARNEY_INF;
  };
  
  bool OrthographicCamera::set1f(const std::string &member, const float &value)
  {
    if (Camera::set1f(member,value))
      return true;
    if (member == "aspect") {
      aspect = value;
      return true;
    }
    if (member == "height") {
      height = value;
      return true;
    }
    if (member == "near") {
      near = value;
      return true;
    }
    if (member == "far") {
      far = value;
      return true;
    }
    return false;
  }
  
  bool OrthographicCamera::set3f(const std::string &member, const vec3f &value)
  {
    if (Camera::set3f(member,value))
      return true;
    if (member == "position") {
      position = value;
      return true;
    }
    if (member == "direction") {
      direction = normalize(value);
      return true;
    }
    if (member == "up") {
      up = value;
      return true;
    }
    return false;
  }
    
  void OrthographicCamera::commit()
  {
    vec3f from   = (const vec3f&)position;
    vec3f dir_00 = normalize(direction);
    
    vec3f dir_du = normalize(cross(dir_00, up));
    vec3f dir_dv = normalize(cross(dir_du, dir_00));

    dd.type = Camera::ORTHOGRAPHIC;
    dd.orthographic.dir     = dir_00;
    dd.orthographic.org_00  = from;
    dd.orthographic.org_du  = dir_du;
    dd.orthographic.org_dv  = dir_dv;
    dd.orthographic.height  = height;
    dd.orthographic.aspect  = aspect;
  }
    

  // ##################################################################
  Camera::SP Camera::create(Context *owner,
                            const std::string &type)
  {
    if (type == "perspective")
      return std::make_shared<PerspectiveCamera>(owner);
    if (type == "orthographic")
      return std::make_shared<OrthographicCamera>(owner);
    
    owner->warn_unsupported_object("Camera",type);
    return {};
  }

}

