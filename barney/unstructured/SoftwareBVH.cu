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

#include "barney/Geometry.h"
#include "barney/DataGroup.h"
#include "cuBQL/bvh.h"

namespace barney {

  struct UMesh_SoftwareBVH {
    
    UMesh_SoftwareBVH(DataGroup *owner,
             const Material &material)
      : Geometry(owner,material)
    {}
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Geometry{}"; }
    
  };
    
}