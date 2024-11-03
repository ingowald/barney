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

#include "barney/render/Renderer.h"
#include "barney/Context.h"

namespace barney {

  /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
  Renderer::Renderer(Context *context) : Object(context) {};

  /*! pretty-printer for printf-debugging */
  std::string Renderer::toString() const
  {
    return "barney::Renderer";
  }

  Renderer::SP Renderer::create(Context *context)
  { return std::make_shared<Renderer>(context); }

  void Renderer::commit() 
  {
    bgColor         = staged.bgColor;
    ambientRadiance = staged.ambientRadiance;
    pathsPerPixel   = staged.pathsPerPixel;
    bgTexture       = staged.bgTexture;
  }
  
  bool Renderer::setObject(const std::string &member,
                           const std::shared_ptr<Object> &value) 
  {
    if (Object::setObject(member,value))
      return true;
    if (member == "bgTexture") {
      staged.bgTexture = value->as<Texture>();
      return true;
    }
    return false;
  }
  
  bool Renderer::set1f(const std::string &member, const float &value)
  {
    if (Object::set1f(member,value))
      return true;
    if (member == "ambientRadiance") {
      staged.ambientRadiance = value;
      return true;
    }
    return false;
  }
  
  bool Renderer::set1i(const std::string &member, const int &value)
  {
    if (Object::set1i(member,value))
      return true;
    if (member == "pathsPerPixel") {
      staged.pathsPerPixel = value;
      return true;
    }
    if (member == "crosshairs") {
      staged.crosshairs = value;
      return true;
    }
    return false;
  }
  
  bool Renderer::set4f(const std::string &member, const vec4f &value)
  {
    if (Object::set4f(member,value))
      return true;
    if (member == "bgColor") {
      staged.bgColor = value;
      return true;
    }
    return false;
  }

  Renderer::DD Renderer::getDD(const Device *device) const
  {
    Renderer::DD dd;
    dd.bgColor = bgColor;
    dd.bgTexture
      = bgTexture ? bgTexture->getTextureObject(device)
      : 0;
    dd.ambientRadiance = ambientRadiance;
    dd.pathsPerPixel = pathsPerPixel;
    return dd;
  }
  
}
