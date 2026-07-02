// SPDX-FileCopyrightText:
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0


#pragma once

#include "barney/amr/BlockStructuredField.h"
#include "barney/volume/MCAccelerator.h"
#include "barney/common/CuBQL.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace BARNEY_NS {

  struct BlockStructuredField;
  
  /*! a block structured amr scalar field, with a CuBQL bvh sampler */
  struct BlockStructuredCuBQLSampler : public ScalarFieldSampler {
    enum { BVH_WIDTH = 4 };
    using bvh_t  = cuBQL::WideBVH<float,3,BVH_WIDTH>;
    using node_t = typename bvh_t::Node;
    
    struct DD : public BlockStructuredField::DD {
#if RTC_DEVICE_CODE
      inline __rtc_device float sample(vec3f P, bool dbg = false) const;
#endif
      bvh_t bvh;
      int useCoarsestSampling = 0;
    };
    DD getDD(Device *device);

    /*! per-device data - parent store the bs-amr field, we just store the
      bvh nodes */
    struct PLD {
      bvh_t bvh = { 0,0 };
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;

    BlockStructuredCuBQLSampler(BlockStructuredField *mesh);
    
    /*! builds the string that allows for properly matching optix
      device progs for this type */
    inline static std::string typeName() { return "BlockStructured_CuBQL"; }

    void build() override;

    BlockStructuredField *const field;
    const DevGroup::SP devices;
    bool useCoarsestSampling = false;
  };
  
  struct BlockStructuredSamplerPTD {
    inline __rtc_device BlockStructuredSamplerPTD(const BlockStructuredCuBQLSampler::DD *field)
      : field(field)
    {}
#if RTC_DEVICE_CODE
    inline __rtc_device void visitBrick(vec3f P, int primID)
    {
      field->addBasisFunctions(sumWeightedValues,sumWeights,primID,P);
    }
#endif
    const BlockStructuredCuBQLSampler::DD *const field;
    
    float sumWeights = 0.f;
    float sumWeightedValues = 0.f;
  };
  
#if RTC_DEVICE_CODE
  inline __rtc_device
  float BlockStructuredCuBQLSampler::DD::sample(vec3f P, bool dbg) const
  {
    if (useCoarsestSampling)
      return sampleBlockStructuredAtCoarsest(*this, P);
    return sampleBlockStructuredAt(*this, P);
  }

  inline __rtc_device vec3f
  amrSampleGrad(const BlockStructuredCuBQLSampler::DD &dd, vec3f P)
  {
    if (dd.useCoarsestSampling)
      return sampleBlockStructuredGradCoarsest(dd, P);
    return sampleBlockStructuredGrad(dd, P);
  }
#endif  
}


