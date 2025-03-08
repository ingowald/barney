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
#if BARNEY_RTC_EMBREE
# include "cuBQL/builder/cpu.h"
#else
# include "cuBQL/builder/cuda.h"
#endif

namespace BARNEY_NS {

  struct UMeshReorderElements {
    Element        *out;
    Element        *in;
    const uint32_t *primIDs;
    int             numElements;
    
    inline __rtc_device void run(const rtc::ComputeInterface &ci)
    {
      int li = ci.launchIndex().x;
      if (li >= numElements) return;

      out[li] = in[primIDs[li]];
    }
  };
  
  UMeshCUBQLSampler::UMeshCUBQLSampler(UMeshField *mesh)
    : mesh(mesh),
      devices(mesh->devices)
  {
    perLogical.resize(devices->numLogical);
  }

  UMeshCUBQLSampler::PLD *UMeshCUBQLSampler::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  
  UMeshCUBQLSampler::DD UMeshCUBQLSampler::getDD(Device *device)
  {
    DD dd;
    (UMeshField::DD &)dd = mesh->getDD(device);
    dd.bvhNodes = getPLD(device)->bvhNodes;
    return dd;
  }

  void UMeshCUBQLSampler::build()
  {
    int numElements = mesh->numElements;
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      if (pld->bvhNodes != 0) {
        /* BVH already built! */
        continue;
      }

      std::cout << "------------------------------------------" << std::endl;
      std::cout << "building UMeshCUBQL BVH!" << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      
      SetActiveGPU forDuration(device);
      
      bvh_t bvh;
      box3f *primBounds
        = (box3f*)device->rtc->allocMem(numElements*sizeof(box3f));
      range1f *valueRanges
        = (range1f*)device->rtc->allocMem(numElements*sizeof(range1f));
      mesh->computeElementBBs(device,
                              primBounds,valueRanges);
      device->rtc->sync();

      
#if BARNEY_RTC_EMBREE
      cuBQL::cpu::spatialMedian(bvh,
                                (const cuBQL::box_t<float,3>*)primBounds,
                                numElements,
                                cuBQL::BuildConfig());
#else
      /*! iw - make sure to have cubql use regular device memory, not
          async mallocs; else we may allocate all memory on the first
          gpu */
      cuBQL::DeviceMemoryResource memResource;
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numElements,
                        cuBQL::BuildConfig(),
                        0,
                        memResource);
#endif
      device->rtc->sync();
      device->rtc->freeMem(primBounds);
      device->rtc->freeMem(valueRanges);
    
      Element *reorderedElements
        = (Element *)device->rtc->allocMem(numElements*sizeof(Element));
      UMeshReorderElements args =
        {
          // Element  *out;
          reorderedElements,
          // Element  *in;
          mesh->getPLD(device)->elements,
          // uint32_t *primIDs;
          bvh.primIDs,
          // int numElements;
          numElements
        };
      int bs = 128;
      int nb = divRoundUp(numElements,bs);
      device->umeshReorderElements->launch(nb,bs,&args);
      device->rtc->copy(mesh->getPLD(device)->elements,
                        reorderedElements,
                        numElements*sizeof(Element));
      device->rtc->sync();
      device->rtc->freeMem(reorderedElements);

      // "save the nodes"
      pld->bvhNodes
        = (node_t *)device->rtc->allocMem(bvh.numNodes*sizeof(node_t));
      device->rtc->copy(pld->bvhNodes,bvh.nodes,bvh.numNodes*sizeof(node_t));
      device->rtc->sync();
      
      // ... and kill whatever else cubql may have in the bvh
#if BARNEY_RTC_EMBREE
      cuBQL::cpu::freeBVH(bvh);
#else
      cuBQL::cuda::free(bvh,0,memResource);
#endif      
      std::cout << OWL_TERMINAL_LIGHT_GREEN
                << "#bn.umesh: cubql bvh built ..."
                << OWL_TERMINAL_DEFAULT << std::endl;
    }
  }
  
  RTC_EXPORT_COMPUTE1D(umeshReorderElements,BARNEY_NS::UMeshReorderElements);
}

