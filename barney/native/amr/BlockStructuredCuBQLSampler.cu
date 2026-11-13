// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/amr/BlockStructuredCuBQLSampler.h"

namespace BARNEY_NS {
  namespace native {
    
    BlockStructuredCuBQLSampler
    ::BlockStructuredCuBQLSampler(BlockStructuredField *field)
      : field(field),
        device(field->device)
    {
      // perLogical.resize(devices->numLogical);
    }

    // BlockStructuredCuBQLSampler::PLD *
    // BlockStructuredCuBQLSampler::getPLD(Device *device) 
    // {
    //   assert(device);
    //   assert(device->contextRank() >= 0);
    //   assert(device->contextRank() < perLogical.size());
    //   return &perLogical[device->contextRank()];
    // }
  
    BlockStructuredCuBQLSampler::DD
    BlockStructuredCuBQLSampler::getDD()
    {
      DD dd;
      (BlockStructuredField::DD &)dd = field->getDD();
      dd.bvh = bvh;
      return dd;
    }

    void BlockStructuredCuBQLSampler::build()
    {
      int numPrims = field->numBlocks;
      if (bvh.nodes != nullptr) {
        /* BVH already built! */
        return;
      }

      std::cout << "------------------------------------------" << std::endl;
      std::cout << "building BlockStructuredCuBQL BVH!" << std::endl;
      std::cout << "------------------------------------------" << std::endl;

      SetActiveGPU forDuration(device);

      box3f *primBounds
        = (box3f*)device->rtc->allocMem(numPrims*sizeof(box3f));
      range1f *valueRanges
        = (range1f*)device->rtc->allocMem(numPrims*sizeof(range1f));
      field->computeElementBBs(device,primBounds,valueRanges);
      device->rtc->sync();
#if BARNEY_RTC_CPU || defined(__HIPCC__)
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
  
