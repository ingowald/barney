// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/common/barney-common.h"
#include "barney/DeviceGroup.h"
#include "barney/volume/MCGrid.h"
#include "barney/volume/Volume.h"
#include "barney/geometry/IsoSurface.h"
#include "barney/volume/DDA.h"
#include "barney/render/World.h"
#include "barney/render/OptixGlobals.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/umesh/common/UMeshField.h"
#if RTC_DEVICE_CODE
# include "rtcore/TraceInterface.h"
#endif

namespace BARNEY_NS {
  using render::Ray;
  using render::DeviceMaterial;
 
  struct IconMultiPassSampler : public ScalarFieldSampler {
    typedef std::shared_ptr<IconMultiPassSampler> SP;
    /* the inherited DD type that the VolumeAccel needs to have in
       order to be able to call sample() on */
    struct DD {
      inline __rtc_device
      float sample(vec3f P, bool dbg) const
      {
        return 0.f;
      };
    };

    /*! the prd we use for the sample/query ray we're tracing */
    struct PRD {
      int primID;
    };
    IconMultiPassSampler(UMeshField *field);
    
    DD getDD(Device *device);
    
    void build() override;

    const DevGroup::SP devices;
  };

  /*! the class/object that does the actual optix launch for (each one
      of) our pass(es) */
  struct IconMultiPassLaunch : MultiPassObject {
    void launch(Device *device,
                const render::World::DD &world,
                const affine3f &instanceXfm,
                render::Ray *rays,
                int numRays) override;
  };

  /*! the 'VolumeAccel' that barney requires each volume to be able to
      create for each instance of a (anari-)volume being created. will
      actually create a IconMultiPassLaumch object */
  struct IconMultiPassAccel : public MCVolumeAccel<IconMultiPassSampler> {
    
    IconMultiPassAccel(Volume *volume,
                       IconMultiPassSampler::SP sampler);
    
    void build(bool full_rebuild) override;
  };

  
}
