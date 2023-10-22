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

#include "barney/Volume.h"
#include "barney/DataGroup.h"

namespace barney {

  struct UMeshField : public ScalarField {
    typedef std::shared_ptr<UMeshField> SP;
    
    UMeshField(DataGroup *owner,
               std::vector<vec4f> &vertices,
               std::vector<TetIndices> &tetIndices,
               std::vector<PyrIndices> &pyrIndices,
               std::vector<WedIndices> &wedIndices,
               std::vector<HexIndices> &hexIndices)
      : ScalarField(owner),
        vertices(std::move(vertices)),
        tetIndices(std::move(tetIndices)),
        pyrIndices(std::move(pyrIndices)),
        wedIndices(std::move(wedIndices)),
        hexIndices(std::move(hexIndices))
    {}

    std::vector<vec4f>      vertices;
    std::vector<TetIndices> tetIndices;
    std::vector<PyrIndices> pyrIndices;
    std::vector<WedIndices> wedIndices;
    std::vector<HexIndices> hexIndices;
  };

}
