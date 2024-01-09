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

#include "barney/volume/StructuredData.h"

namespace barney {

  StructuredData::StructuredData(const vec3i &dims,
                                 BNScalarType scalarType,
                                 const void *scalars,
                                 const vec3f &gridOrigin,
                                 const vec3f &gridSpacing)
    : dims(dims),
      scalarType(scalarType),
      rawScalarData(scalars),
      gridOrigin(gridOrigin),
      gridSpacing(gridSpacing)
  {}
}

