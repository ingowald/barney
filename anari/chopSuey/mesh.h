// ======================================================================== //
// Copyright 2022-2022 Stefan Zellmann                                      //
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

// std
#include <memory>
#include <string>
#include <vector>
// anari
#include "anari/anari_cpp/ext/linalg.h"
// ours
#include "box3.h"

namespace chop {

  using float3 = anari::math::float3;
  using int3 = anari::math::int3;
  using box3 = anari::math::box3;

  struct Geometry {
    typedef std::shared_ptr<Geometry> SP;

    std::vector<float3> vertex;
    std::vector<int3> index;
  };

  struct Mesh {
    typedef std::shared_ptr<Mesh> SP;

    box3 bounds;
    std::vector<Geometry::SP> geoms;

    static Mesh::SP loadOBJ(std::string fileName);
    static Mesh::SP loadMini(std::string fileName);
    static Mesh::SP load(std::string fileName);
  };

}
