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

namespace barney {

  Camera::Camera(Context *owner)
    : Object(owner)
  {}

  // ##################################################################

  struct PerspectiveCamera : public Camera {
    PerspectiveCamera(Context *owner)
      : Camera(owner)
    {
      char *fl = getenv("BARNEY_FOCAL_LENGTH");
      if (fl) defaultValues.focalLength = std::stof(fl);
      char *lr = getenv("BARNEY_LENS_RADIUS");
      if (lr) defaultValues.lensRadius = std::stof(lr);
    }
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
  };
  
  bool PerspectiveCamera::set1f(const std::string &member, const float &value)
  {
    if (Camera::set1f(member,value))
      return true;
    if (member == "aspect") {
      aspect = value;
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
    
    vec3f dir_du = //aspect *
      normalize(cross(dir_00, up));
    vec3f dir_dv = normalize(cross(dir_du, dir_00));

    dir_00 *= (float)(1.f / (2.0f * tanf((0.5f * fovy) * (float)M_PI / 180.0f)));
    // dir_00 -= 0.5f * dir_du;
    // dir_00 -= 0.5f * dir_dv;

    dd.dir_00 = (float3&)dir_00;
    dd.dir_du = (float3&)dir_du;
    dd.dir_dv = (float3&)dir_dv;
    dd.lens_00 = (float3&)from;

    dd.lensRadius  = defaultValues.lensRadius;
    dd.focalLength = defaultValues.focalLength;
  }
    

  // ##################################################################
  Camera::SP Camera::create(Context *owner,
                            const std::string &type)
  {
    if (type == "perspective")
      return std::make_shared<PerspectiveCamera>(owner);
    
    owner->warn_unsupported_object("Camera",type);
    return {};
  }

}

