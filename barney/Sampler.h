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

#include "barney/Object.h"
#include "barney/material/DeviceSampler.h"

namespace barney {

  struct Sampler {
    typedef std::shared_ptr<Sampler> SP;
    virtual void create(render::DeviceSampler &dd, int devID) = 0;
    int   samplerID = -1;
    int   inAttribute { render::ATTRIBUTE_0 };
    mat4f outTransform { mat4f::identity() };
    vec4f outOffset { 0.f, 0.f, 0.f, 0.f };
  };
  struct TransformSampler : public Sampler {
    void create(render::DeviceSampler &dd, int devID) override;
  };
  struct ImageSampler : public Sampler {
    void create(render::DeviceSampler &dd, int devID) override;
    
    mat4f inTransform { mat4f::identity() };
    vec4f inOffset { 0.f, 0.f, 0.f, 0.f };
    Texture::SP image{ 0 };
  };
  
}

