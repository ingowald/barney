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

#include "barney.h"
#include "barney/core/common/common.h"

#define WARN_NOTIMPLEMENTED std::cout << " ## " << __PRETTY_FUNCTION__ << " not implemented yet ..." << std::endl;

namespace barney {

  BN_API
  BNModel bnModelCreate(BNContext ctx,
                        int dataGroupIdx,
                        BNGroup *groups,
                        int numGroups,
                        BNVolume *volumes,
                        int numVolumes)
  {
    WARN_NOTIMPLEMENTED;
    return 0;
  }

  BN_API
  void bnContextDestroy(BNContext context)
  {
    WARN_NOTIMPLEMENTED;
  }

  BN_API
  BNGeom bnTriangleMeshCreate(BNContext context,
                              int dataGroupIdx,
                              BNMaterial *material,
                              int3 *indices,
                              int numIndices,
                              float3 *vertices,
                              int numVertices,
                              float3 *normals,
                              float2 *texcoords)
  {
    return 0;
  }

  BN_API
  BNGroup bnGroupCreate(BNContext context,
                        int dataGroupIdx,
                        BNGeom *geoms, int numGeoms,
                        BNVolume *volumes, int numVolumes)
  {
    return 0;
  }
  
}
