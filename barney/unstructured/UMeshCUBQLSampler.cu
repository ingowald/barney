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

#include "barney/unstructured/UMeshCUBQLSampler.h"
#include "cuBQL/bvh.h"

namespace barney {

  UMeshCUBQLSampler::UMeshCUBQLSampler(ScalarField *field)
    : mesh((UMeshField *)field)
  {}
  
  void UMeshCUBQLSampler::build()
  {
    BARNEY_CUDA_SYNC_CHECK();
    assert(mesh);
    assert(!mesh->elements.empty());
    
    if (bvhNodesBuffer != 0) {
      std::cout << "cubql bvh already built..." << std::endl;
      return;
    }
    auto devGroup = mesh->devGroup;

    bvh_t bvh;

    // this initially builds ONLY on first GPU!
    box3f *d_primBounds = 0;
    BARNEY_CUDA_SYNC_CHECK();
    BARNEY_CUDA_CALL(MallocManaged(&d_primBounds,mesh->elements.size()*sizeof(box3f)));
    
    BARNEY_CUDA_SYNC_CHECK();

    auto d_mesh = mesh->getDD(0);
    computeElementBoundingBoxes
      <<<divRoundUp((int)mesh->elements.size(),1024),1024>>>
      (d_primBounds,d_mesh);
    BARNEY_CUDA_SYNC_CHECK();
    
    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = 8;
    buildConfig.enableSAH();
    static cuBQL::ManagedMemMemoryResource managedMem;
    cuBQL::gpuBuilder(bvh,
                      (const cuBQL::box_t<float,3>*)d_primBounds,
                      (uint32_t)mesh->elements.size(),
                      buildConfig,
                      (cudaStream_t)0,
                      managedMem);
    std::vector<Element> reorderedElements(mesh->elements.size());
    for (int i=0;i<mesh->elements.size();i++) {
      reorderedElements[i] = mesh->elements[bvh.primIDs[i]];
    }
    mesh->elements = reorderedElements;
    owlBufferUpload(mesh->elementsBuffer,reorderedElements.data());
    BARNEY_CUDA_CALL(Free(d_primBounds));

    bvhNodesBuffer
      = owlDeviceBufferCreate(devGroup->owl,OWL_USER_TYPE(node_t),
                              bvh.numNodes,bvh.nodes);
    // primIDsBuffer
    //   = owlDeviceBufferCreate(devGroup->owl,OWL_INT,
    //                           bvh.numPrims,bvh.primIDs);
    cuBQL::free(bvh,0,managedMem);
  }
  
}
