// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
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

#include "barney/unstructured/SharedFacesSampler.h"

namespace barney {
  /*! sort indices of the edge A-B such that A<B; reverse orientation
      if this requires a swap */
  void sortIndices(int &A, int &B, int &orientation)
  {
    if (A > B) {
      std::swap(A,B);
      orientation = 1-orientation;
    }
  }
  
  /*! sort indices of the tet face such that A<B<C; reverse
      orientation if this requires swapping the face orientation */
  void sortIndices(vec3i &face, int &orientation)
  {
    sortIndices(face.y,face.z,orientation);
    sortIndices(face.x,face.y,orientation);
    sortIndices(face.y,face.z,orientation);
  };
  
  template<typename Lambda>
  void iterateFaces(umesh::Tet tet, Lambda lambda)
  {
    int A = tet.x;
    int B = tet.y;
    int C = tet.z;
    int D = tet.w;
    std::vector<vec3i> faces = {
                                vec3i{ A, C, B },
                                vec3i{ A, D, C },
                                vec3i{ A, B, D },
                                vec3i{ B, C, D }
    };
    for (auto face : faces) {
      int orientation = 0;
      sortIndices(face,orientation);
      lambda(face,orientation);
    }
  }
}
