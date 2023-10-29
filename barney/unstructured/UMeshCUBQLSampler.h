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

   /*! can sample umeshes using cuda-point location of cuBQL-bvh. this
       samples a *field* and thus doesn't know anything about transfer
       functions or acceleration */
  struct UMeshCUBQLSampler {
    using bvh_t  = cuBQL::BinaryBVH<float,3>;
    using node_t = typename bvh_t::Node;
    
    struct DD {
      /*! sample the umesh field; can return NaN if sample did not hit
        any unstructured element at all */
      inline __device__ float sample(vec3f P, bool dbg=false) const;

      bvh_t          bvh;
      UMeshField::DD mesh;
    };
    
    UMeshCUBQLSampler(ScalarField *field);
    void build();

    UMeshField *const mesh;
    OWLBuffer   bvhNodesBuffer = 0;
    OWLBuffer   primIDsBuffer  = 0;
  };

  /*! sample the umesh field; can return NaN if sample did not hit
    any unstructured element at all */
  inline __device__
  float UMeshCUBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    // return .5f;
    int nodeStack[30];
    int *stackPtr = nodeStack;
    *stackPtr++ = 0;
    if (dbg) printf("sample %f %f %f\n",P.x,P.y,P.z);
    while (stackPtr > nodeStack) {
      node_t node = bvh.nodes[*--stackPtr];
      if (!((const box3f&)node.bounds).contains(P))
        continue;
      if (node.count == 0) {
        *stackPtr++ = node.offset + 0;
        *stackPtr++ = node.offset + 1;
      } else {
        int primID = bvh.primIDs[node.offset];
        auto idx = mesh.tetIndices[primID];
        auto vtx = mesh.vertices[idx.x];
        if (dbg) {
          printf("primID %i/%i -> %i %i %i %i vtx %f %f %f %f\n",primID,int(stackPtr-nodeStack),
                 idx.x,idx.y,idx.z,idx.w,
                 vtx.x,vtx.y,vtx.z,vtx.w
                 );
        }
        return vtx.w;
      }
    }
  }
  
}
