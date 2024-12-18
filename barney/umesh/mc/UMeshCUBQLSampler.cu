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

#include "barney/umesh/mc/UMeshCUBQLSampler.h"

namespace barney {

  const std::vector<Device::SP> &UMeshCUBQLSampler::Host::getDevices()
  {
    assert(mesh);
    return mesh->getDevices();
  }

  
  DevGroup *UMeshCUBQLSampler::Host::getDevGroup()
  {
    assert(mesh);
    return mesh->getDevGroup();
  }

  void UMeshCUBQLSampler::Host::build(bool full_rebuild)
  {
    if (bvhNodesBuffer) {
      return;
    }

    auto dev = getDevices()[0];
    assert(dev);
    SetActiveGPU forDuration(dev);

    BARNEY_CUDA_SYNC_CHECK();
    assert(mesh);
    assert(!mesh->elements.empty());

    if (bvhNodesBuffer != 0) {
      return;
    }
    auto devGroup = getDevGroup();

    bvh_t bvh;

    // this initially builds ONLY on first GPU!
    box3f *d_primBounds = 0;
    BARNEY_CUDA_SYNC_CHECK();
    BARNEY_CUDA_CALL(MallocManaged(&d_primBounds,
                                   mesh->elements.size()*sizeof(box3f)));
    BARNEY_CUDA_SYNC_CHECK();

    std::cout << OWL_TERMINAL_BLUE
              << "#bn.umesh: computing umesh element BBs ..."
              << OWL_TERMINAL_DEFAULT << std::endl;
    mesh->computeElementBBs(dev,///*deviceID:*/0,
                            d_primBounds);
    BARNEY_CUDA_SYNC_CHECK();
    
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.umesh: building cubql bvh ..."
              << OWL_TERMINAL_DEFAULT << std::endl;
    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = 3;
#if BARNEY_CUBQL_HOST
    cuBQL::cpu::spatialMedian(bvh,
                               (const cuBQL::box_t<float,3>*)d_primBounds,
                               (uint32_t)mesh->elements.size(),
                               buildConfig);

#else
    static cuBQL::ManagedMemMemoryResource managedMem;
    cuBQL::gpuBuilder(bvh,
                      (const cuBQL::box_t<float,3>*)d_primBounds,
                      (uint32_t)mesh->elements.size(),
                      buildConfig,
                      (cudaStream_t)0,
                      managedMem);
#endif
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.umesh: cubql bvh built ..."
              << OWL_TERMINAL_DEFAULT << std::endl;
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
#if BARNEY_CUBQL_HOST
    cuBQL::cpu::freeBVH(bvh);
#else
    cuBQL::cuda::free(bvh,0,managedMem);
#endif
    std::cout << OWL_TERMINAL_LIGHT_GREEN
              << "#bn.umesh: cubql bvh built ..."
              << OWL_TERMINAL_DEFAULT << std::endl;
  }
  
}
