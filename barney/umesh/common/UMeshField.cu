// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/common/barney-common.h"
#include "barney/umesh/common/UMeshField.h"
#include "barney/Context.h"
#include "barney/umesh/mc/UMeshCUBQLSampler.h"
#include "barney/volume/MCGrid.cuh"
#include "barney/umesh/os/AWT.h"
#if RTC_DEVICE_CODE
# include "rtcore/TraceInterface.h"
#endif

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/UMeshMC,/*name*/UMeshMC,
                       /*geomtype device data */
                       MCVolumeAccel<UMeshCUBQLSampler>::DD,false,false);
  
  UMeshField::PLD *UMeshField::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }

#if RTC_DEVICE_CODE
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
      od1 = a;
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
#endif
  
  struct UMeshRasterElements {
    /* kernel data */
    UMeshField::DD mesh;
    MCGrid::DD     grid;

#if RTC_DEVICE_CODE
    inline __rtc_device
    void run(const rtc::ComputeInterface &ci);
#endif
  };

#if RTC_DEVICE_CODE
  inline __rtc_device
  void UMeshRasterElements::run(const rtc::ComputeInterface &ci)
  {
    const int eltIdx = ci.launchIndex().x;
    if (eltIdx >= mesh.numElements) return;    

    auto elt = mesh.elements[eltIdx];
    if (elt.type == Element::TET) {
      const vec4i indices = *(const vec4i *)&mesh.indices[elt.ofs0];
      vec4f a = rtc::load(mesh.vertices[indices.x]);
      vec4f b = rtc::load(mesh.vertices[indices.y]);
      vec4f c = rtc::load(mesh.vertices[indices.z]);
      vec4f d = rtc::load(mesh.vertices[indices.w]);
      rasterTet<5>(grid,a,b,c,d);
    } else {
      const box4f eltBounds = mesh.eltBounds(elt);
      rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
    }
  }
#endif

  /*! KERNEL that creates an elements[] array from the
      elementOffsets[] array */
  struct UMeshCreateElements {
    /*! kernel ARGS */
    UMeshField::DD dd;
    int      numIndices;
    int     *elementOffsets;
    box3f   *d_worldBounds;

#if RTC_DEVICE_CODE
    inline __rtc_device void run(const rtc::ComputeInterface &ci);
#endif
  };

#if RTC_DEVICE_CODE
    /*! kernel CODE */
  inline __rtc_device void UMeshCreateElements::run(const rtc::ComputeInterface &ci)
    {
      int tid = ci.launchIndex().x;
      if (tid >= dd.numElements) return;

      int begin = elementOffsets[tid];
      int end
        = (tid == (dd.numElements-1))
        ? numIndices
        : elementOffsets[tid+1];
      Element &elt = ((Element *)dd.elements)[tid];

      elt.ofs0 = begin;
      switch (end-begin) {
      case 4:
        elt.type = Element::TET;
        break;
      case 5:
        elt.type = Element::PYR;
        break;
      case 6:
        elt.type = Element::WED;
        break;
      case 8:
        elt.type = Element::HEX;
        break;
      default:
        printf("@bn.umesh: invalid element with indices range [%i...%i)\n",
               begin,end);
      }

      box4f bounds = dd.eltBounds(elt);

      rtc::fatomicMin(&d_worldBounds->lower.x,bounds.lower.x);
      rtc::fatomicMin(&d_worldBounds->lower.y,bounds.lower.y);
      rtc::fatomicMin(&d_worldBounds->lower.z,bounds.lower.z); 
      rtc::fatomicMax(&d_worldBounds->upper.x,bounds.upper.x);
      rtc::fatomicMax(&d_worldBounds->upper.y,bounds.upper.y);
      rtc::fatomicMax(&d_worldBounds->upper.z,bounds.upper.z); 
      // if (tid < 10) {
      //   printf("(%i) bounds (%f %f %f)(%f %f %f)->(%f %f %f)(%f %f %f)\n",
      //          tid,
      //          bounds.lower.x,
      //          bounds.lower.y,
      //          bounds.lower.z,
      //          bounds.upper.x,
      //          bounds.upper.y,
      //          bounds.upper.z,
      //          d_worldBounds->lower.x,
      //          d_worldBounds->lower.y,
      //          d_worldBounds->lower.z,
      //          d_worldBounds->upper.x,
      //          d_worldBounds->upper.y,
      //          d_worldBounds->upper.z);
      // }
    }
#endif
  
  void UMeshField::buildMCs(MCGrid &grid)
  {
    buildInitialMacroCells(grid);
  }
  
  /*! build *initial* macro-cell grid (ie, the scalar field min/max
    ranges, but not yet the majorants) over a umesh */
  void UMeshField::buildInitialMacroCells(MCGrid &grid)
  {
    if (grid.built()) {
      // initial grid already built
      return;
    }

    // std::cout << "------------------------------------------" << std::endl;
    // std::cout << "rebuilding ENTIRE mc grid!!!!" << std::endl;
    // std::cout << "------------------------------------------" << std::endl;
    
    float maxWidth = reduce_max(worldBounds.size());//getBox(worldBounds).size());
    int MC_GRID_SIZE
      = 200 + int(sqrtf(elementOffsets->count/100.f));
    vec3i dims = 1+vec3i(worldBounds.size() * ((MC_GRID_SIZE-1) / maxWidth));
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.um: building initial macro cell grid of " << dims << " MCs"
              << OWL_TERMINAL_DEFAULT << std::endl;
    grid.resize(dims);
    
    grid.gridOrigin
      = worldBounds.lower;
    grid.gridSpacing
      = worldBounds.size() * rcp(vec3f(dims));
    
    grid.clearCells();
    
    const int bs = 128;
    const int nb = divRoundUp(numElements,bs);
    for (auto device : *devices) {
      UMeshRasterElements args = {
        getDD(device),
        grid.getDD(device)
      };
      device->umeshRasterElements->launch(nb,bs,&args);
    }
    for (auto device : *devices)
      device->sync();
  }
    
  
  /*! computes - ON CURRENT DEVICE - the given mesh's prim bounds and
    per-prim scalar ranges, and writes those into givne
    pre-allocated device mem location */
  struct UMeshComputeElementBBs {
    /* kernel ARGS */
    box3f         *d_primBounds;
    range1f       *d_primRanges;
    UMeshField::DD mesh;

#if RTC_DEVICE_CODE
    inline __rtc_device
    void run(const rtc::ComputeInterface &ci);
#endif
  };

#if RTC_DEVICE_CODE
    /* kernel FUNCTION */
    inline __rtc_device
    void UMeshComputeElementBBs::run(const rtc::ComputeInterface &ci)
    {
      const int tid = ci.launchIndex().x;
      if (tid >= mesh.numElements) return;
      
      auto elt = mesh.elements[tid];
      box4f eb = mesh.eltBounds(elt);
      d_primBounds[tid] = getBox(eb);
      if (d_primRanges) d_primRanges[tid] = getRange(eb);
    }
#endif
  
  bool UMeshField::setData(const std::string &member,
                           const std::shared_ptr<Data> &value)
  {
    if (ScalarField::setData(member,value)) return true;

    if (member == "elementOffsets") {
      elementOffsets = value->as<PODData>();
      return true;
    }
    if (member == "vertices") {
      vertices = value->as<PODData>();
      return true;
    }
    if (member == "indices") {
      indices = value->as<PODData>();
      return true;
    }

    return false;
  }
    
  
  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitmives ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeaable on specified device */
  void UMeshField::computeElementBBs(Device  *device,
                                     box3f   *d_primBounds,
                                     range1f *d_primRanges)
  {
    UMeshComputeElementBBs args = {
      /* kernel ARGS */
      // box3f         *d_primBounds;
      d_primBounds,
      // range1f       *d_primRanges;
      d_primRanges,
      // UMeshField::DD mesh;
      getDD(device)
    };
    int bs = 128;
    int nb = divRoundUp(numElements,bs);
    device->umeshComputeElementBBs->launch(nb,bs,&args);
    device->sync();
  }

  UMeshField::UMeshField(Context *context, 
                         const DevGroup::SP   &devices)
    : ScalarField(context,devices)
  {
    perLogical.resize(devices->numLogical);
  }
  
  void UMeshField::commit()
  {
    assert(indices);
    assert(vertices);
    assert(elementOffsets);
    
    std::cout << "#bn.umesh: computing device-side elements[] representation"
              << std::endl;
    this->numElements = (int)elementOffsets->count;
    for (auto device : *devices) {
      PLD *pld = getPLD(device); 
      auto rtc = device->rtc;
      SetActiveGPU forDuration(device);
      
      int numElements = (int)elementOffsets->count;
      int numIndices  = (int)indices->count;
      if (pld->elements)
        rtc->freeMem(pld->elements);
      if (pld->pWorldBounds)
        rtc->freeMem(pld->pWorldBounds);
      
      pld->pWorldBounds
        = (box3f*)rtc->allocMem(sizeof(box3f));
      pld->elements
        = (Element*)rtc->allocMem(numElements*sizeof(Element));

      box3f emptyBox;
      rtc->copy(pld->pWorldBounds,&emptyBox,sizeof(emptyBox));
      UMeshCreateElements args = {
        // UMeshField::DD dd;
        // int      numIndices;
        // int     *elementOffsets;
        // box3f   *d_worldBounds;
        getDD(device),
        numIndices,
        (int*)elementOffsets->getDD(device),
        pld->pWorldBounds
      };
      int bs = 128;
      int nb = divRoundUp(numElements,bs);
      device->umeshCreateElements->launch(nb,bs,&args);
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
    }
  }
  

  UMeshField::DD UMeshField::getDD(Device *device)
  {
    assert(device);
    UMeshField::DD dd;
    assert(vertices);
    assert(indices);
    assert(getPLD(device)->elements);
    (ScalarField::DD &)dd = ScalarField::getDD(device);
    dd.vertices = (rtc::float4 *)vertices->getDD(device);
    dd.indices  = (int    *)indices->getDD(device);
    dd.elements = (Element*)getPLD(device)->elements;
    dd.numElements = (int)elementOffsets->count;
    assert(dd.vertices);
    assert(dd.indices);
    assert(dd.elements);
    return dd;
  }
  
  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
#if 0
    return std::make_shared<AWTAccel>(volume,this);
#else
    auto sampler
      = std::make_shared<UMeshCUBQLSampler>(this);
    return std::make_shared<MCVolumeAccel<UMeshCUBQLSampler>>
      (volume,
       createGeomType_UMeshMC,
       sampler);
#endif
  }

  RTC_EXPORT_COMPUTE1D(umeshRasterElements,UMeshRasterElements);
  RTC_EXPORT_COMPUTE1D(umeshCreateElements,UMeshCreateElements);
  RTC_EXPORT_COMPUTE1D(umeshComputeElementBBs,UMeshComputeElementBBs);
}


