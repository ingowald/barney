// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#if RTC_DEVICE_CODE
# include "rtcore/TraceInterface.h"
#endif

namespace BARNEY_NS {
  using render::Ray;
  using render::DeviceMaterial;
 
  template<typename SFSampler>
  struct MCVolumeAccel : public VolumeAccel 
  {
    struct DD 
    {
      Volume::DD<SFSampler> volume;
      MajorantsGrid::DD     mcGrid;
    };

    struct PLD {
      rtc::Geom  *geom  = 0;
      rtc::Group *group = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank()]; } 
    std::vector<PLD> perLogical;

    DD getDD(Device *device)
    {
      DD dd;
      dd.volume = volume->getDD(device,sfSampler);
      dd.mcGrid = majorantsGrid->getDD(device);
      return dd;
    }

    MCVolumeAccel(Volume *volume,
                  GeomTypeCreationFct creatorFct,
                  const std::shared_ptr<SFSampler> &sfSampler);

      GeomTypeCreationFct const creatorFct;
    
    void build(bool full_rebuild) override;

#if BARNEY_DEVICE_PROGRAM
    /*! optix bounds prog for this class of accels */
    static inline __rtc_device
    void boundsProg(const rtc::TraceInterface &ti,
                    const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __rtc_device
    void isProg(rtc::TraceInterface &ti);
#endif 
    
    MajorantsGrid::SP majorantsGrid;
    const std::shared_ptr<SFSampler> sfSampler;
  };
  
  template<typename SFSampler>
  struct MCIsoSurfaceAccel : public IsoSurfaceAccel 
  {
    struct DD 
    {
      IsoSurface::DD<SFSampler> isoSurface;
      MCGrid::DD mcGrid;
    };
    
    struct PLD {
      rtc::Geom  *geom  = 0;
      rtc::Group *group = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank()]; } 
    std::vector<PLD> perLogical;
    
    DD getDD(Device *device)
    {
      DD dd;
      dd.isoSurface = isoSurface->getDD(device,sfSampler);
      // dd.sf     = sfSampler->getDD(device);
      dd.mcGrid = mcGrid->getDD(device);
      return dd;
    }
    
    MCIsoSurfaceAccel(IsoSurface *isoSurface,
                      GeomTypeCreationFct creatorFct,
                      const std::shared_ptr<SFSampler> &sfSampler);
    
    GeomTypeCreationFct const creatorFct;
    
    void build() override;
    
#if BARNEY_DEVICE_PROGRAM
    /*! optix bounds prog for this class of accels */
    static inline __rtc_device
    void boundsProg(const rtc::TraceInterface &ti,
                    const void *geomData,
                    owl::common::box3f &bounds,
                    const int32_t primID);
    /*! optix isec prog for this class of accels */
    static inline __rtc_device
    void isProg(rtc::TraceInterface &ti);
#endif 
    
    MCGrid::SP       mcGrid;
    const std::shared_ptr<SFSampler> sfSampler;
  };
  
  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  template<typename SFSampler>
  void MCVolumeAccel<SFSampler>::build(bool full_rebuild) 
  {
    if (!majorantsGrid) {
      auto mcGrid = volume->sf->getMCs();
      majorantsGrid = std::make_shared<MajorantsGrid>(mcGrid);
    }
    majorantsGrid->computeMajorants(&volume->xf);
    sfSampler->build();
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFct);
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
        pld->group = device->rtc->createUserGeomsGroup({geom});
      }
      pld->group->buildAccel();
      
      // now let the actual volume we're building know about the
      // group we just created
      Volume::PLD *volumePLD = volume->getPLD(device);
      if (volumePLD->generatedGroups.empty()) 
        volumePLD->generatedGroups = { pld->group };
    }
  }



  template<typename SFSampler>
  void MCIsoSurfaceAccel<SFSampler>::build() 
  {
    mcGrid = isoSurface->sf->getMCs();
    sfSampler->build();
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      
      // build our own internal per-device data: one geom, and one
      // group that contains it.
      PLD *pld = getPLD(device);
      if (!pld->geom) {
        rtc::GeomType *gt
          = device->geomTypes.get(creatorFct);
        // build a single-prim geometry, that single prim is our
        // entire MC/DDA grid
        pld->geom = gt->createGeom();
        pld->geom->setPrimCount(1);
      }
      rtc::Geom *geom = pld->geom;
      DD dd = getDD(device);
      geom->setDD(&dd);
      
      IsoSurface::PLD *isoSurfacePLD = isoSurface->getPLD(device);
      isoSurfacePLD->userGeoms = { geom };
      // if (!pld->group) {
      //   // now put that into a instantiable group, and build it.
      //   pld->group = device->rtc->createUserGeomsGroup({geom});
      // }
      // pld->group->buildAccel();

      // // now let the actual volume we're building know about the
      // // group we just created
      // IsoSurface::PLD *isoSurfacePLD = isoSurface->getPLD(device);
      // if (isoSurfacePLD->generatedGroups.empty()) 
      //   isoSurfacePLD->generatedGroups = { pld->group };
    }
  }
  
  

  template<typename SFSampler>
  MCVolumeAccel<SFSampler>::
  MCVolumeAccel(Volume *volume,
                GeomTypeCreationFct creatorFct,
                const std::shared_ptr<SFSampler> &sfSampler)
    : VolumeAccel(volume),
      sfSampler(sfSampler),
      creatorFct(creatorFct)
  {
    perLogical.resize(devices->numLogical);
  }

  template<typename SFSampler>
  MCIsoSurfaceAccel<SFSampler>::
  MCIsoSurfaceAccel(IsoSurface *isoSurface,
                    GeomTypeCreationFct creatorFct,
                    const std::shared_ptr<SFSampler> &sfSampler)
    : IsoSurfaceAccel(isoSurface),
      sfSampler(sfSampler),
      creatorFct(creatorFct)
  {
    perLogical.resize(devices->numLogical);
  }
  
  // ------------------------------------------------------------------
  // device progs: macro-cell accel with DDA traversal
  // ------------------------------------------------------------------

#if BARNEY_DEVICE_PROGRAM && RTC_DEVICE_CODE
  template<typename SFSampler>
  inline __rtc_device
  void MCVolumeAccel<SFSampler>::boundsProg(const rtc::TraceInterface &ti,
                                            const void *geomData,
                                            owl::common::box3f &bounds,
                                            const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.volume.sfCommon.worldBounds;
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCIsoSurfaceAccel<SFSampler>::boundsProg(const rtc::TraceInterface &ti,
                                                const void *geomData,
                                                owl::common::box3f &bounds,
                                                const int32_t primID)
  {
    const DD &self = *(DD*)geomData;
    bounds = self.isoSurface.sfCommon.worldBounds;
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCIsoSurfaceAccel<SFSampler>::isProg(rtc::TraceInterface &ti)
  {
    const void *pd = ti.getProgramData();
           
    const DD &self = *(typename MCIsoSurfaceAccel<SFSampler>::DD*)pd;
    const render::World::DD &world = render::OptixGlobals::get(ti).world;
    // ray in world space
    Ray &ray = *(Ray*)ti.getPRD();
#ifdef NDEBUG
    const bool dbg = false;
#else
    const bool dbg = ray.dbg();
#endif
    
    box3f bounds = self.isoSurface.sfCommon.worldBounds;
    range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
    
    // ray in object space
    vec3f obj_org = ti.getObjectRayOrigin();
    vec3f obj_dir = ti.getObjectRayDirection();

    if (dbg) {
      printf("MCIsoAccel isec %f %f %f mcgrid %i %i %i\n",
             obj_dir.x,
             obj_dir.y,
             obj_dir.z,
             self.mcGrid.dims.x,
             self.mcGrid.dims.y,
             self.mcGrid.dims.z
             );
    }
    
    auto objRay = ray;
    objRay.org = obj_org;
    objRay.dir = obj_dir;

    if (!boxTest(objRay,tRange,bounds))
      return;

    // ------------------------------------------------------------------
    // compute ray in macro cell grid space 
    // ------------------------------------------------------------------
    vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    vec3f dda_org = obj_org;
    vec3f dda_dir = obj_dir;

    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

#if 1
    Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
                                ti.getGeometryIndex(),
                                ti.getPrimitiveIndex()));
#else
    Random rng(ray.rngSeed.next(hash(ti.getRTCInstanceIndex(),
                                     ti.getGeometryIndex(),
                                     ti.getPrimitiveIndex())));
#endif
    
    float tHit = ray.tMax;
    dda::dda3(dda_org,dda_dir,tRange.upper,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
                float _t0 = t0;
                float _t1 = t1;
                range1f tRange = range1f {t0,min(t1,ray.tMax)};
                if (tRange.lower >= tRange.upper) return true;
                
                range1f valueRange = self.mcGrid.scalarRange(cellIdx);
                
                // scalar values at begin/end of current ray segment
                // (NOT sorted by value as valuerange is!)
                float ff0 = 0.f, ff1 = 0.f;
                if (dbg) printf("dda %i %i %i [%f %f] -> [%f %f]\n",
                                cellIdx.x,
                                cellIdx.y,
                                cellIdx.z,
                                tRange.lower,
                                tRange.upper,
                                valueRange.lower,
                                valueRange.upper);
                auto overlaps = [&](float isoValue)
                {
                  return
                    isoValue >= valueRange.lower &&
                    isoValue <= valueRange.upper;
                };
                if (!overlaps(self.isoSurface.isoValue))
                  return true;

                auto intersect = [&](float isoValue)
                {
                  float t = (isoValue - ff0) / (ff1-ff0);
                  t = lerp_l(t,tRange.lower,tRange.upper);
                  tHit = min(tHit,t);
                };
                  

                float tt1 = t0;
                vec3f P = obj_org + tt1 * obj_dir;
                ff1 = self.isoSurface.sfSampler.sample(P,dbg);
                int numSteps = 10; 
                for (int i=1;i<=numSteps;i++) {
                  float tt0 = tt1;
                  ff0 = ff1;
                  tt1 = lerp_l(i/float(numSteps),_t0,_t1);
                  P = obj_org + tt1 * obj_dir;
                  ff1 = self.isoSurface.sfSampler.sample(P,dbg);
                  if (isnan(ff0) || isnan(ff1)) continue;
                  
                  valueRange.lower = min(ff0,ff1);
                  valueRange.upper = max(ff0,ff1);
                  tRange = range1f{tt0,tt1};
                  
                  if (dbg)
                    printf(" ... t [%f %f] v [ %f %f ]\n",
                           tRange.lower,
                           tRange.upper,
                           valueRange.lower,
                           valueRange.upper);
                  if (overlaps(self.isoSurface.isoValue)) {
                    intersect(self.isoSurface.isoValue);
                    if (tHit < ray.tMax) {
                      return false;
                    }
                  }
                }

                return true;
              },
              /*NO debug:*/false
              );
    if (tHit >= ray.tMax) return;
    
    // ------------------------------------------------------------------
    // get texture coordinates
    // ------------------------------------------------------------------
    const vec3f osP  = obj_org + tHit * obj_dir;
    vec3f P  = ti.transformPointFromObjectToWorldSpace(osP);
#if 1
    float delta
      = length(bounds.size()) * .1f
      / float(self.mcGrid.dims.x+self.mcGrid.dims.y+self.mcGrid.dims.z);
    
    float fP   = self.isoSurface.sfSampler.sample(osP);
    float fPx0 = self.isoSurface.sfSampler.sample(osP+vec3f(-delta,0.f,0.f));
    float fPx1 = self.isoSurface.sfSampler.sample(osP+vec3f(+delta,0.f,0.f));
    float fPy0 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,-delta,0.f));
    float fPy1 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,+delta,0.f));
    float fPz0 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,0.f,-delta));
    float fPz1 = self.isoSurface.sfSampler.sample(osP+vec3f(0.f,0.f,+delta));
    float dx = 2.f;
    float dy = 2.f;
    float dz = 2.f;
    if (isnan(fPx0)) { dx -= 1.f; fPx0 = fP; }
    if (isnan(fPx1)) { dx -= 1.f; fPx1 = fP; }
    if (isnan(fPy0)) { dy -= 1.f; fPy0 = fP; }
    if (isnan(fPy1)) { dy -= 1.f; fPy1 = fP; }
    if (isnan(fPz0)) { dz -= 1.f; fPz0 = fP; }
    if (isnan(fPz1)) { dz -= 1.f; fPz1 = fP; }
    vec3f osN(dx == 0.f ? 0.f : (fPx1-fPx0) / dx,
              dy == 0.f ? 0.f : (fPy1-fPy0) / dy,
              dz == 0.f ? 0.f : (fPz1-fPz0) / dz);
    if (osN == vec3f(0.f))
      osN = -normalize(obj_dir);
    vec3f n = ti.transformNormalFromObjectToWorldSpace(osN);
#else
    vec3f osN = - normalize(obj_dir);
    vec3f n   = - normalize(obj_dir);
#endif
    int primID    = ti.getPrimitiveIndex();
    int instID    = ti.getInstanceID();
    
    render::HitAttributes hitData;
    hitData.worldPosition   = ti.transformPointFromObjectToWorldSpace(osP);
    hitData.worldNormal     = normalize(n);
    hitData.objectPosition  = osP;
    hitData.objectNormal    = make_vec4f(normalize(osN));
    hitData.primID          = primID;
    hitData.instID          = instID;
    hitData.t               = tHit;
    hitData.isShadowRay     = ray.isShadowRay;
    float u = 0.f;
    float v = 0.f;
    auto interpolator
      = [u,v,dbg](const GeometryAttribute::DD &attrib,
                  bool faceVarying) -> vec4f
      {
        return vec4f(1.f);
      };
    self.isoSurface.setHitAttributes(hitData,interpolator,world,dbg);

    const DeviceMaterial &material
      = world.materials[self.isoSurface.materialID];
      
    PackedBSDF bsdf
      = material.createBSDF(hitData,world.samplers,dbg);
    float opacity
      = bsdf.getOpacity(ray.isShadowRay,ray.isInMedium,
                        ray.dir,hitData.worldNormal,ray.dbg());
    if (opacity < 1.f) {
      if (rng() > opacity) {
        return;
      }
    }
    material.setHit(ray,hitData,world.samplers,dbg);

    // Write hit IDs for AOV channels
    const render::OptixGlobals &globals = render::OptixGlobals::get(ti);
    if (globals.hitIDs) {
      const int rayID
        = ti.getLaunchIndex().x
        + ti.getLaunchDims().x
        * ti.getLaunchIndex().y;
      if (tHit < globals.hitIDs[rayID].depth) {
        globals.hitIDs[rayID].primID = primID;
        globals.hitIDs[rayID].instID
          = globals.world.instIDToUserInstID
          ? globals.world.instIDToUserInstID[instID]
          : instID;
        globals.hitIDs[rayID].objID  = self.isoSurface.userID;
        globals.hitIDs[rayID].depth  = tHit;
      }
    }
  }
  
  template<typename SFSampler>
  inline __rtc_device
  void MCVolumeAccel<SFSampler>::isProg(rtc::TraceInterface &ti)
  {
    const void *pd = ti.getProgramData();
           
    const DD &self = *(typename MCVolumeAccel<SFSampler>::DD*)pd;
    const render::World::DD &world = render::OptixGlobals::get(ti).world;
    // ray in world space
    Ray &ray = *(Ray*)ti.getPRD();
#ifdef NDEBUG
    enum { dbg = false };
#else
    const bool dbg = ray.dbg();
#endif
    
    box3f bounds = self.volume.sfCommon.worldBounds;
    range1f tRange = { ti.getRayTmin(), ti.getRayTmax() };
    
    // ray in object space
    vec3f obj_org = ti.getObjectRayOrigin();
    vec3f obj_dir = ti.getObjectRayDirection();

    auto objRay = ray;
    objRay.org = obj_org;
    objRay.dir = obj_dir;

    if (!boxTest(objRay,tRange,bounds))
      return;
    
    // ------------------------------------------------------------------
    // compute ray in macro cell grid space 
    // ------------------------------------------------------------------
    vec3f mcGridOrigin  = self.mcGrid.gridOrigin;
    vec3f mcGridSpacing = self.mcGrid.gridSpacing;

    vec3f dda_org = obj_org;
    vec3f dda_dir = obj_dir;

    dda_org = (dda_org - mcGridOrigin) * rcp(mcGridSpacing);
    dda_dir = dda_dir * rcp(mcGridSpacing);

    Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
                                ti.getGeometryIndex(),0));
    // Random rng(ray.rngSeed,hash(ti.getRTCInstanceIndex(),
    //                             ti.getGeometryIndex(),
    //                             ti.getPrimitiveIndex()));
    dda::dda3(dda_org,dda_dir,tRange.upper,
              vec3ui(self.mcGrid.dims),
              [&](const vec3i &cellIdx, float t0, float t1) -> bool
              {
                const float majorant = self.mcGrid.majorant(cellIdx);
                
                if (majorant == 0.f) return true;
                
                vec4f   sample = 0.f;
                range1f tRange = {t0,min(t1,ray.tMax)};
                if (!Woodcock::sampleRange(sample,
                                           self.volume,
                                           obj_org,
                                           obj_dir,
                                           tRange,
                                           majorant,
                                           rng,
                                           dbg)) 
                  return true;
                if (dbg) printf("woodcock hit sample %f %f %f:%f\n",
                                sample.x,
                                sample.y,
                                sample.z,
                                sample.w);
                
                vec3f P_obj = obj_org + tRange.upper * obj_dir;
                vec3f P = ti.transformPointFromObjectToWorldSpace(P_obj);
                ray.setVolumeHit(P,
                                 tRange.upper,
                                 getPos(sample));

                // Write hit IDs for AOV channels on first non-transparent voxel
                const render::OptixGlobals &globals = render::OptixGlobals::get(ti);
                if (globals.hitIDs) {
                  const int rayID
                    = ti.getLaunchIndex().x
                    + ti.getLaunchDims().x
                    * ti.getLaunchIndex().y;
                  if (tRange.upper < globals.hitIDs[rayID].depth) {
                    globals.hitIDs[rayID].primID = ti.getPrimitiveIndex();
                    globals.hitIDs[rayID].instID
                      = globals.world.instIDToUserInstID
                      ? globals.world.instIDToUserInstID[ti.getInstanceID()]
                      : ti.getInstanceID();
                    globals.hitIDs[rayID].objID  = self.volume.userID;
                    globals.hitIDs[rayID].depth  = tRange.upper;
                  }
                }

                ti.reportIntersection(tRange.upper, 0);
                return false;
              },
              /*NO debug:*/false
              );
  }
#endif
}
