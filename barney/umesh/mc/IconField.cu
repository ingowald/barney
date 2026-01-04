// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/umesh/mc/IconField.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  
  RTC_IMPORT_TRIANGLES_GEOM(/*file*/IconField,/*name*/IconField,
                            /*geomtype device data */
                            IconMultiPassSampler::DD,false,true);
  
  void IconMultiPassAccel::build(bool full_rebuild)
  {
    if (!majorantsGrid) {
      auto mcGrid = volume->sf->getMCs();
      majorantsGrid = std::make_shared<MajorantsGrid>(mcGrid);
    }
    majorantsGrid->computeMajorants(&volume->xf);
    sfSampler->build();

    auto thisPass = std::make_shared<IconMultiPassLaunch>();
    volume->generatedPasses = { thisPass };
    
#if 0
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);

      auto creatorFunction = createGeomType_IconField;
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFunction);
        // build a single-prim geometry, that single prim is our
        // entire MC/DDA grid
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
      }
      rtc::Geom *geom = pld->geom;
      DD dd = getDD(device);
      geom->setDD(&dd);
      
      if (!pld->group) {
        // now put that into a instantiable group, and build it.
        pld->group = device->rtc->createTrianglesGroup({geom});
        // pld->group = device->rtc->createUserGeomsGroup({geom});
      }
      pld->group->buildAccel();
      
      // now let the actual volume we're building know about the
      // group we just created
      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty()) 
        volumePLD->generatedGroups = { pld->group };
    }
#endif
  }

  void IconMultiPassLaunch::trace(const render::World *world,
                                  const affine3f &instanceXfm,
                                  render::Ray *rays,
                                  int numRays)
  {
    PING;
  }

  IconMultiPassAccel::IconMultiPassAccel(Volume *volume,
                                         IconMultiPassSampler::SP sampler)
    : MCVolumeAccel<IconMultiPassSampler>(volume,nullptr,sampler)
  {
    PING;
  }

  IconMultiPassSampler::IconMultiPassSampler(UMeshField *field)
  {
    PING;
  }

    
  IconMultiPassSampler::DD IconMultiPassSampler::getDD(Device *device)
  {
    PING;
    return {};
  }

    
  void IconMultiPassSampler::build()
  {
    PING;
  }


}
