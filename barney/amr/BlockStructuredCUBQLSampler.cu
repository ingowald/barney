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

#include "barney/amr/BlockStructuredCUBQLSampler.h"

namespace barney {

  void BlockStructuredCUBQLSampler::Host::build(bool full_rebuild)
  {
    PING;
    if (bvhNodesBuffer) {
      std::cout <<" bvh already built" << std::endl;
      return;
    }
    
    auto devGroup = field->getDevGroup();
    SetActiveGPU forDuration(devGroup->devices[0]);
    
    BARNEY_CUDA_SYNC_CHECK();
    
    if (bvhNodesBuffer != 0) {
      std::cout << "cubql bvh already built..." << std::endl;
      return;
    }

    bvh_t bvh;

    // this initially builds ONLY on first GPU!
    box3f *d_primBounds = 0;
    BARNEY_CUDA_SYNC_CHECK();
    BARNEY_CUDA_CALL(MallocManaged(&d_primBounds,
                                   field->blockIDs.size()*sizeof(box3f)));
    BARNEY_CUDA_SYNC_CHECK();

    field->computeBlockFilterDomains(/*deviceID:*/0,d_primBounds);
    BARNEY_CUDA_SYNC_CHECK();
    
    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = 7;
    static cuBQL::ManagedMemMemoryResource managedMem;
    cuBQL::gpuBuilder(bvh,
                      (const cuBQL::box_t<float,3>*)d_primBounds,
                      (uint32_t)field->blockIDs.size(),
                      buildConfig,
                      (cudaStream_t)0,
                      managedMem);
    std::vector<uint32_t> reorderedElements(field->blockIDs.size());
    for (int i=0;i<field->blockIDs.size();i++) {
      reorderedElements[i] = field->blockIDs[bvh.primIDs[i]];
    }
    field->blockIDs = reorderedElements;
    owlBufferUpload(field->blockIDsBuffer,reorderedElements.data());
    BARNEY_CUDA_CALL(Free(d_primBounds));

    bvhNodesBuffer
      = owlDeviceBufferCreate(devGroup->owl,OWL_USER_TYPE(node_t),
                              bvh.numNodes,bvh.nodes);
    cuBQL::free(bvh,0,managedMem);
    std::cout << "cubql bvh built ..." << std::endl;
  }
  
}
