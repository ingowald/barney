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

#include "barney/unstructured/UMeshMCAccelerator.h"

namespace barney {
  template<typename VolumeSampler>
  void UMeshMCAccelerator<VolumeSampler>::buildMCs() 
  {
    assert(mesh);
    assert(volume);
    PING;
    mesh->buildInitialMacroCells(mcGrid);
    PRINT(mcGrid.dims);
    PING;
    mcGrid.computeMajorants(&volume->xf);
    PING;
  }

  template<typename VolumeSampler>
  void UMeshMCAccelerator<VolumeSampler>::build()
  {
    BARNEY_CUDA_SYNC_CHECK();
    buildMCs();
    BARNEY_CUDA_SYNC_CHECK();
    this->sampler.build();
    BARNEY_CUDA_SYNC_CHECK();
  }

  // template struct UMeshMCAccelerator<UMeshQCSampler>;
  template struct UMeshMCAccelerator<UMeshCUBQLSampler>;
}

