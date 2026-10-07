// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Splats.h"

namespace BARNEY_NS {
  namespace splats {
    namespace native {
      
      Splats::Splats(native::Context *context, int slot)
        : Geometry(context,context->getDevices(slot))
      {
        perLogical.resize(devices->numLogical);
      }

      Splats::~Splats()
      {
        free();
      }
      
      Splats::PLD *Splats::getPLD(Device *device)
      {
        assert(device);
        assert(device->contextRank() >= 0);
        assert(device->contextRank() < perLogical.size());
        return &perLogical[device->contextRank()];
      }
      
        
      /*! pretty-printer for printf-debugging */
      std::string Splats::toString() const
      {
        return "barney::samplePlugins::native::Splats";
      }
      
      void Splats::free()
      {
        for (auto device : *devices) {
          SetActiveGPU forDuration(device);
          PLD *pld = getPLD(device);
#if BARNEY_RTC_CPU
          if (pld->bvh.nodes) cuBQL::cpu::freeBVH(pld->bvh);
#else

          if (pld->bvh.nodes) cuBQL::cuda::free(pld->bvh);
#endif
        }
      }

    __rtc_global 
    void computeElementBBs(rtc::ComputeInterface ci,
                           box3f *primBounds,
                           int numSplats,
                           int *indices,
                           vec3f *positions,
                           float *radii,
                           float globalRadius)
    {
#if RTC_DEVICE_CODE
      const int tid = ci.launchIndex().x;
      if (tid >= numSplats) return;

      int idx = (indices?indices[tid]:tid);
      vec3f pos = positions[idx];
      float radius = radii?radii[idx]:globalRadius;
      box3f bb(pos-radius,pos+radius);
      primBounds[tid] = bb;
#endif
    }
      
      void Splats::build()
      {
        int numSplats
          = indexData
          ? indexData->count
          : vertexPositionData->count;
        
        for (auto device : *devices) {
          PLD *pld = getPLD(device);
          if (pld->bvh.nodes != 0) {
            /* BVH already built! */
            continue;
          }
          SetActiveGPU forDuration(device);
          
          box3f *primBounds
            = (box3f*)device->rtc->allocMem(numSplats*sizeof(box3f));
          range1f *valueRanges
            = (range1f*)device->rtc->allocMem(numSplats*sizeof(range1f));
          __rtc_launch(device->rtc,computeElementBBs,
                       divRoundUp(numSplats,128),128,
                       primBounds,
                       numSplats,
                       (int*)(indexData?indexData->getDD(device):nullptr),
                       (vec3f*)vertexPositionData->getDD(device),
                       (float*)(vertexRadiusData?vertexRadiusData->getDD(device):nullptr),
                       globalRadius);
          device->rtc->sync();
#if BARNEY_RTC_CPU
          cuBQL::cpu::spatialMedian(pld->bvh,
                                    (const cuBQL::box_t<float,3>*)primBounds,
                                    numSplats,
                                    cuBQL::BuildConfig());
#else
          cuBQL::DeviceMemoryResource memResource;
          cuBQL::gpuBuilder(pld->bvh,
                            (const cuBQL::box_t<float,3>*)primBounds,
                            numSplats,
                            cuBQL::BuildConfig(),
                            0,
                            memResource);
#endif
          device->rtc->sync();
          device->rtc->freeMem(primBounds);
          device->rtc->freeMem(valueRanges);
        }
      }

      
    }
  }
}
