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

#include "barney/DataGroup.h"
#include "barney/volume/MCAccelerator.h"

namespace barney {

  /*! a structured 3D scalar field; strictly speaking in barney this
      isn't actually a "volume" (it only becomes a volume after paired
      with a transfer function), but this name still sounds more
      reasonable thatn `Structured3DScalarField`, which would be a
      more accurate name */
  struct StructuredVolume : public ScalarField
  {
    struct DD {
      cudaTextureObject_t texture;
    };
    
    StructuredVolume(const affine3f &unitCellToWorldTransform,
                     BNScalarType scalarType,
                     const void *scalars,
                     size_t sizeScalars);

    const BNScalarType   scalarType;
    const affine3f       unitCellToWorldTransform;
    std::vector<uint8_t> rawScalarData;
  };
}

