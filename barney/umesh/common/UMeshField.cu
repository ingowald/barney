// SPDX-FileCopyrightText:
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0

#include "barney/common/barney-common.h"
#include "barney/umesh/common/UMeshField.h"
#include "barney/Context.h"
#include "barney/umesh/mc/UMeshCuBQLSampler.h"
#include "barney/volume/MCGrid.cuh"
// #include "barney/umesh/os/AWT.h"
#if RTC_DEVICE_CODE
# include "rtcore/ComputeInterface.h"
# include "rtcore/TraceInterface.h"
#endif

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/UMeshMC,/*name*/UMeshMC,
                       /*geomtype device data */
                       MCVolumeAccel<UMeshCuBQLSampler>::DD,false,false);
  RTC_IMPORT_USER_GEOM(/*file*/UMeshMC,/*name*/UMeshMC_Iso,
                       /*geomtype device data */
                       MCIsoSurfaceAccel<UMeshCuBQLSampler>::DD,false,false);
  
  UMeshField::PLD *UMeshField::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  inline __rtc_device float length3(vec4f v)
  { return length(getPos(v)); }
  
  template<int D> inline __rtc_device
  void rasterTet(MCGrid::DD grid,
                 vec4f a,
                 vec4f b,
                 vec4f c,
                 vec4f d)
  {
    float lab = length3(b-a);
    float lac = length3(c-a);
    float lad = length3(d-a);
    float lbc = length3(c-b);
    float lbd = length3(d-b);
    float lcd = length3(d-c);
    float maxLen = max(max(max(max(max(lab,lac),lad),lbc),lbd),lcd);

    vec4f ab = 0.5f*(a+b);
    vec4f ac = 0.5f*(a+c);
    vec4f ad = 0.5f*(a+d);
    vec4f bc = 0.5f*(b+c);
    vec4f bd = 0.5f*(b+d);
    vec4f cd = 0.5f*(c+d);

    vec4f oa,ob,oc,od0,od1;
    if (lab >= maxLen) {
      oa = ab;
      ob = c;
      oc = d;
      od0 = a;
      od1 = b;
    } else if (lac >= maxLen) {
      oa = ac;
      ob = b;
      oc = d;
      od0 = a;
      od1 = c;
    } else if (lad >= maxLen) {
      oa = ad;
      ob = b;
      oc = c;
      od0 = a;
      od1 = d;
    } else if (lbc >= maxLen) {
      oa = bc;
      ob = a;
      oc = d;
      od0 = b;
      od1 = c;
    } else if (lbd >= maxLen) {
      oa = bd;
      ob = a;
      oc = c;
      od0 = b;
      od1 = d;
    } else {
      oa = cd;
      ob = a;
      oc = b;
      od0 = c;
      od1 = d;
    }
    rasterTet<D-1>(grid,oa,ob,oc,od0);
    rasterTet<D-1>(grid,oa,ob,oc,od1);
  }
  
  template<> inline __rtc_device
  void rasterTet<0>(MCGrid::DD grid,
                    vec4f a,
                    vec4f b,
                    vec4f c,
                    vec4f d)
  {
    box4f bb;
    bb.extend(a);
    bb.extend(b);
    bb.extend(c);
    bb.extend(d);
    rasterBox(grid,bb);
  }
  
  __rtc_global void umeshRasterCells(rtc::ComputeInterface ci,
                                     UMeshField::DD mesh,
                                     MCGrid::DD grid)
  {
    const int cellIdx = ci.launchIndex().x;
    if (cellIdx >= mesh.numCells) return;    

    auto cellType = mesh.cellTypes[cellIdx];
    if (cellType == _VTK_TET || cellType == _ANARI_TET) {
      uint32_t ofs0 = mesh.cellOffsets[cellIdx];
      const vec4i indices = *(const vec4i *)&mesh.indices[ofs0];
      int sidx_x = mesh.scalarsArePerVertex ? indices.x : cellIdx;
      int sidx_y = mesh.scalarsArePerVertex ? indices.y : cellIdx;
      int sidx_z = mesh.scalarsArePerVertex ? indices.z : cellIdx;
      int sidx_w = mesh.scalarsArePerVertex ? indices.w : cellIdx;
      vec4f a(mesh.vertices[indices.x],mesh.scalars[sidx_x]);
      vec4f b(mesh.vertices[indices.y],mesh.scalars[sidx_y]);
      vec4f c(mesh.vertices[indices.z],mesh.scalars[sidx_z]);
      vec4f d(mesh.vertices[indices.w],mesh.scalars[sidx_w]);
      rasterTet<5>(grid,a,b,c,d);
    } else {
      const box4f eltBounds = mesh.cellBounds(cellIdx);
      rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
    }
  }
  
  MCGrid::SP UMeshField::buildMCs()
  {
    MCGrid::SP mcGrid = std::make_shared<MCGrid>(devices);
    buildInitialMacroCells(*mcGrid);
    return mcGrid;
  }
  
  /*! build *initial* macro-cell grid (ie, the scalar field min/max
    ranges, but not yet the majorants) over a umesh */
  void UMeshField::buildInitialMacroCells(MCGrid &grid)
  {
    if (grid.built()) {
      // initial grid already built
      return;
    }
    assert(!worldBounds.empty());

    float maxWidth = reduce_max(worldBounds.size());//getBox(worldBounds).size());
    int MC_GRID_SIZE
      = 200 + int(sqrtf(cellOffsets->count/1000.f));
    vec3i dims = 1+vec3i(worldBounds.size() * ((MC_GRID_SIZE-1) / maxWidth));
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.um: building initial macro cell grid of " << dims << " MCs"
              << OWL_TERMINAL_DEFAULT << std::endl;
    grid.resize(dims);
    
    grid.gridOrigin  = worldBounds.lower;
    grid.gridSpacing = worldBounds.size() * rcp(vec3f(dims));
    
    grid.clearCells();
    
    for (auto device : *devices) 
      __rtc_launch(device->rtc,
                   umeshRasterCells,
                   divRoundUp(numCells,128),128,
                   getDD(device),grid.getDD(device));
    for (auto device : *devices)
      device->sync();
  }
    
  bool UMeshField::setData(const std::string &member,
                           const std::shared_ptr<Data> &value)
  {
    if (ScalarField::setData(member,value))
      return true;

    if (member == "cell.index") {
      cellOffsets = value->as<PODData>();
      return true;
    }
    if (member == "cell.type") {
      cellTypes = value->as<PODData>();
      return true;
    }
    if (member == "cell.data") {
      scalars = value->as<PODData>();
      scalarsArePerVertex = false;
      return true;
    }
    if (member == "vertex.position") {
      vertices = value->as<PODData>();
      return true;
    }
    if (member == "vertex.data") {
      scalars = value->as<PODData>();
      scalarsArePerVertex = true;
      return true;
    }
    if (member == "index") {
      indices = value->as<PODData>();
      return true;
    }

    return false;
  }
    
  __rtc_global 
  void umeshComputeElementBBs(rtc::ComputeInterface ci,
                              UMeshField::DD mesh,
                              box3f   *d_primBounds,
                              range1f *d_primRanges)
  {
    const int tid = ci.launchIndex().x;
    if (tid >= mesh.numCells) return;
    
    box4f eb = mesh.cellBounds(tid);
    d_primBounds[tid] = getBox(eb);
    if (d_primRanges) d_primRanges[tid] = getRange(eb);
  }
  
  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitmives ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeaable on specified device */
  void UMeshField::computeElementBBs(Device  *device,
                                     box3f   *d_primBounds,
                                     range1f *d_primRanges)
  {
    __rtc_launch(device->rtc,umeshComputeElementBBs,
                 divRoundUp(numCells,128),128,
                 getDD(device),d_primBounds,d_primRanges);
    device->rtc->sync();
  }

  
  UMeshField::UMeshField(Context *context, 
                         const DevGroup::SP   &devices)
    : ScalarField(context,devices)
  {
    perLogical.resize(devices->numLogical);
  }

  __rtc_global
  void computeWorldBounds(rtc::ComputeInterface ci,
                          UMeshField::DD mesh,
                          box3f *pWorldBounds)
  {
    int tid = ci.launchIndex().x;
    if (tid >= mesh.numCells)
      return;
    box4f bounds = mesh.cellBounds(tid);
    rtc::fatomicMin(&pWorldBounds->lower.x,bounds.lower.x);
    rtc::fatomicMin(&pWorldBounds->lower.y,bounds.lower.y);
    rtc::fatomicMin(&pWorldBounds->lower.z,bounds.lower.z); 
    rtc::fatomicMax(&pWorldBounds->upper.x,bounds.upper.x);
    rtc::fatomicMax(&pWorldBounds->upper.y,bounds.upper.y);
    rtc::fatomicMax(&pWorldBounds->upper.z,bounds.upper.z); 
  }
  
  void UMeshField::commit()
  {
    assert(indices);
    assert(vertices);
    assert(cellOffsets);
    assert(cellTypes);
    assert(scalars);
    
    this->numCells = (int)cellOffsets->count;
    for (auto device : *devices) {
      PLD *pld = getPLD(device); 
      auto rtc = device->rtc;
      SetActiveGPU forDuration(device);

      int numCells = (int)cellOffsets->count;
      assert(numCells > 0);
      int numIndices  = (int)indices->count;
      if (!pld->pWorldBounds) {
        pld->pWorldBounds
          = (box3f*)rtc->allocMem(sizeof(box3f));
      }
      box3f emptyBox;
      rtc->copy(pld->pWorldBounds,&emptyBox,sizeof(emptyBox));
      __rtc_launch(rtc,computeWorldBounds,
                   divRoundUp(numCells,128),128,
                   getDD(device),pld->pWorldBounds);
    }
    
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      device->sync();
      // in case of having multiple devices this will repeately
      // download the same value; that's ok.
      PLD *pld = getPLD(device); 
      device->rtc->copy(&worldBounds,
                        pld->pWorldBounds,
                        sizeof(worldBounds));
      device->sync();
      assert(!worldBounds.empty());
    }
  }
  

  UMeshField::DD UMeshField::getDD(Device *device)
  {
    assert(device);
    UMeshField::DD dd;
    assert(vertices);
    assert(indices);
    (ScalarField::DD &)dd = ScalarField::getDD(device);
    dd.vertices    = (const vec3f *)vertices->getDD(device);
    dd.scalars     = (const float *)scalars->getDD(device);
    dd.indices     = (const int   *)indices->getDD(device);
    dd.cellOffsets = (const int   *)cellOffsets->getDD(device);
    dd.cellTypes   = (const uint8_t *)cellTypes->getDD(device);
    dd.scalarsArePerVertex = scalarsArePerVertex;
    dd.numCells    = (int)cellOffsets->count;
    assert(dd.numCells > 0);
    assert(dd.vertices);
    assert(dd.indices);
    return dd;
  }
  
  IsoSurfaceAccel::SP UMeshField::createIsoAccel(IsoSurface *isoSurface) 
  {
    auto sampler = std::make_shared<UMeshCuBQLSampler>(this);
    return std::make_shared<MCIsoSurfaceAccel<UMeshCuBQLSampler>>
      (isoSurface,
       createGeomType_UMeshMC_Iso,
       sampler);
  }
  
  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
#if 0
    return std::make_shared<AWTAccel>(volume,this);
#else
    auto sampler
      = std::make_shared<UMeshCuBQLSampler>(this);
    return std::make_shared<MCVolumeAccel<UMeshCuBQLSampler>>
      (volume,
       createGeomType_UMeshMC,
       sampler);
#endif
  }
}


