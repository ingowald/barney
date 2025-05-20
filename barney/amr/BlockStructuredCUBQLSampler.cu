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
#if BARNEY_RTC_EMBREE || defined(__HIPCC__)
# include "cuBQL/builder/cpu.h"
#else
# include "cuBQL/builder/cuda.h"
#endif
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  BlockStructuredCUBQLSampler::BlockStructuredCUBQLSampler(BlockStructuredField *field)
    : field(field),
      devices(field->devices)
  {
    perLogical.resize(devices->numLogical);
  }

  BlockStructuredCUBQLSampler::PLD *BlockStructuredCUBQLSampler::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }
  
  BlockStructuredCUBQLSampler::DD BlockStructuredCUBQLSampler::getDD(Device *device)
  {
    DD dd;
    (BlockStructuredField::DD &)dd = field->getDD(device);
    dd.bvh = getPLD(device)->bvh;
    return dd;
  }

  void BlockStructuredCUBQLSampler::build()
  {
    int numPrims = field->numBlocks;
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      bvh_t &bvh = pld->bvh;
      if (bvh.nodes != nullptr) {
        /* BVH already built! */
        continue;
      }

      std::cout << "------------------------------------------" << std::endl;
      std::cout << "building BlockStructuredCUBQL BVH!" << std::endl;
      std::cout << "------------------------------------------" << std::endl;

      SetActiveGPU forDuration(device);

      box3f *primBounds
        = (box3f*)device->rtc->allocMem(numPrims*sizeof(box3f));
      range1f *valueRanges
        = (range1f*)device->rtc->allocMem(numPrims*sizeof(range1f));
      field->computeElementBBs(device,primBounds,valueRanges);
      device->rtc->sync();
#if BARNEY_RTC_EMBREE || defined(__HIPCC__)
      cuBQL::cpu::spatialMedian(bvh,
                                (const cuBQL::box_t<float,3>*)primBounds,
                                numPrims,
                                cuBQL::BuildConfig());
#else
      /*! make sure to have cubql use regular device memory, not async
        mallocs; else we may allocate all memory on the first gpu */
      cuBQL::DeviceMemoryResource memResource;
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        cuBQL::BuildConfig(),
                        0,
                        memResource);
#endif
      device->rtc->sync();
      device->rtc->freeMem(primBounds);
      device->rtc->freeMem(valueRanges);
    
      std::cout << OWL_TERMINAL_LIGHT_GREEN
                << "#bn.bsfield: cubql bvh built ..."
                << OWL_TERMINAL_DEFAULT << std::endl;
    }
  }
  
}
  
