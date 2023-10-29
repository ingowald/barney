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
#include "barney/unstructured/UMeshField.h"
#include "cuBQL/bvh.h"

namespace barney {

   /*! can sample umeshes using cuda-point location of cuBQL-bvh */
  struct UMeshCUBQLSampler {
    using bvh_t  = cuBQL::BinaryBVH<float,3>;
    using node_t = typename bvh_t::Node;
    
    struct DD {
      /*! sample the umesh field; can return NaN if sample did not hit
        any unstructured element at all */
      inline __device__ float sample(vec3f P);

      UMeshField::DD mesh;
      bvh_t          bvh;
    };
    
    UMeshCUBQLSampler(ScalarField *field);
    void build();

    UMeshField *const mesh;
    OWLBuffer   bvhNodesBuffer = 0;
    OWLBuffer   primIDsBuffer  = 0;
  };
  
}
