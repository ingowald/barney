// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/umesh/mc/IconField.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  using render::OptixGlobals;
  
  RTC_IMPORT_TRIANGLES_GEOM(/*file*/IconField,/*name*/IconField,
                            /*geomtype device data */
                            IconMultiPassSampler::DD,false,true);
  RTC_IMPORT_TRACE2D
  (/*IconField.cu*/IconField,
   /*ray gen name */traceRays_IconField,
   /*launch params data type*/sizeof(BARNEY_NS::render::OptixGlobals)
   );
  
  
  void IconMultiPassAccel::build(bool full_rebuild)
  {
    if (!majorantsGrid) {
      auto mcGrid = volume->sf->getMCs();
      majorantsGrid = std::make_shared<MajorantsGrid>(mcGrid);
    }
    majorantsGrid->computeMajorants(&volume->xf);
    sfSampler->build();

    auto thisPass = std::make_shared<IconMultiPassLaunch>
      (sampler);
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

  void IconMultiPassLaunch::launch(Device *device,
                                   const render::World::DD &world,
                                   const affine3f &instanceXfm,
                                   render::Ray *rays,
                                   int numRays)
  {
    // PING;
    int bs = 128;
    int nb = divRoundUp(numRays,bs);
    auto rayGen = sampler->getPLD(device)->rayGen;

    OptixGlobals dd;
    dd.world = world;
    dd.rays = rays;
    dd.numRays = numRays;
    dd.accel = sampler->getPLD(device)->baseTrisTLAS->getDD();
    rayGen->launch(/* bs,nb intentionally inverted:
                      always have 1024 in width: */
                   vec2i(bs,nb),
                   &dd);
  }

  IconMultiPassAccel::IconMultiPassAccel(Volume *volume,
                                         IconMultiPassSampler::SP sampler)
    : MCVolumeAccel<IconMultiPassSampler>(volume,nullptr,sampler),
      sampler(sampler)
  {
    PING;
  }





  
  IconMultiPassSampler::IconMultiPassSampler(UMeshField *field)
    : perLogical(field->devices->size()),
      field(field)
  {
    PING;
  }

  IconMultiPassSampler::PLD *IconMultiPassSampler::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  IconMultiPassSampler::DD IconMultiPassSampler::getDD(Device *device)
  {
    PING;
    return { getPLD(device)->baseTrisTLAS->getDD() };
  }
    
  void IconMultiPassSampler::build()
  {
    PING;

    // purely for testing:
    box3f bounds = field->worldBounds;
    PRINT(bounds);

    vec3f v0 = bounds.lower;
    vec3f v1 = bounds.upper;
    vec3f v2 = {
      bounds.lower.x,
      bounds.lower.y,
      bounds.upper.z
    };
    vec3f vtx[3] = { v0,v1,v2 };
    vec3i idx(0,1,2);

    auto creatorFunction = createGeomType_IconField;
    for (auto device : *field->devices) {
      auto rtc = device->rtc;
      PLD *pld = getPLD(device);
      if (!pld->baseTrisTLAS) {
        // create a rtc group (ie tlas) for the given object that we
        // can trace rays against, over a single triangle mesh
        rtc::Buffer *vertices = rtc->createBuffer(3*sizeof(vec3f),vtx);
        rtc::Buffer *indices = rtc->createBuffer(1*sizeof(vec3i),&idx);
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFunction);
        rtc::Geom *geom
          = gt->createGeom();
        geom->setPrimCount(1);
        geom->setVertices(vertices, 3);
        geom->setIndices(indices, 1);
        rtc::Group *blas = rtc->createTrianglesGroup({geom});
        blas->buildAccel();
        rtc::Group *tlas = rtc->createInstanceGroup({blas},{},{});
        tlas->buildAccel();
        
        pld->baseTrisTLAS = tlas;

        
        pld->rayGen = createTrace_traceRays_IconField(device->rtc);
      }
    }
  }


}
