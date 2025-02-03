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

#include "barney/umesh/common/UMeshField.h"
#include "barney/Context.h"
#include "barney/umesh/mc/UMeshCUBQLSampler.h"
#include "barney/volume/MCGrid.cuh"

namespace barney {

  UMeshField::PLD *UMeshField::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  
  inline __both__ float length3(vec4f v)
  { return length(getPos(v)); }
  
  template<int D> inline __both__
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
  
  template<> inline __both__
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
  
  struct UMeshRasterElements {
    /* kernel data */
    UMeshField::DD mesh;
    MCGrid::DD     grid;
    
    template<typename CI>
    inline __both__
    void run(const CI &ci)
    {
      const int eltIdx = ci.launchIndex().x;
      if (eltIdx >= mesh.numElements) return;    

      auto elt = mesh.elements[eltIdx];
      if (elt.type == Element::TET) {
        const vec4i indices = *(const vec4i *)&mesh.indices[elt.ofs0];
        vec4f a = make_vec4f(mesh.vertices[indices.x]);
        vec4f b = make_vec4f(mesh.vertices[indices.y]);
        vec4f c = make_vec4f(mesh.vertices[indices.z]);
        vec4f d = make_vec4f(mesh.vertices[indices.w]);
        rasterTet<5>(grid,a,b,c,d);
      } else {
        const box4f eltBounds = mesh.eltBounds(elt);
        rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
      }
    }
  };

  /*! KERNEL that creates an elements[] array from the
      elementOffsets[] array */
  struct UMeshCreateElements {
    /*! kernel ARGS */
    UMeshField::DD dd;
    int      numIndices;
    int     *elementOffsets;
    box3f   *d_worldBounds;

    /*! kernel CODE */
    template<typename CI>
    inline __both__ void run(const CI &ci)
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
      barney::fatomicMin(&d_worldBounds->lower.x,bounds.lower.x);
      barney::fatomicMin(&d_worldBounds->lower.y,bounds.lower.y);
      barney::fatomicMin(&d_worldBounds->lower.z,bounds.lower.z); 
      barney::fatomicMax(&d_worldBounds->upper.x,bounds.upper.x);
      barney::fatomicMax(&d_worldBounds->upper.y,bounds.upper.y);
      barney::fatomicMax(&d_worldBounds->upper.z,bounds.upper.z); 
    }
  };
  
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

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "rebuilding ENTIRE mc grid!!!!" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    float maxWidth = reduce_max(worldBounds.size());//getBox(worldBounds).size());
    int MC_GRID_SIZE
      = 200 + int(sqrtf(elementOffsets->count/10));
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
    
    /* kernel FUNCTION */
    template<typename ComputeInterface>
    inline __both__
    void run(const ComputeInterface &ci)
    {
      const int tid = ci.launchIndex().x;
      if (tid >= mesh.numElements) return;
      
      auto elt = mesh.elements[tid];
      box4f eb = mesh.eltBounds(elt);
      d_primBounds[tid] = getBox(eb);
      if (d_primRanges) d_primRanges[tid] = getRange(eb);
    }
  };

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
// #if 1
//     BARNEY_NYI();
// #else
//     assert(device);
//     SetActiveGPU forDuration(device);
//     int bs = 1024;
//     int nb = divRoundUp(int(elements.size()),bs);
//     CHECK_CUDA_LAUNCH(g_computeElementBoundingBoxes,
//                       nb,bs,0,0,
//                       d_primBounds,d_primRanges,getDD(device));
//     // g_computeElementBoundingBoxes
//     //   <<<nb,bs>>>(d_primBounds,d_primRanges,getDD(device));
//     BARNEY_CUDA_SYNC_CHECK();
// #endif
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
    this->numElements = elementOffsets->count;
    for (auto device : *devices) {
      PLD *pld = getPLD(device); 
      auto rtc = device->rtc;
      
      int numElements = elementOffsets->count;
      int numIndices  = indices->count;
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
    
    for (auto device : *devices)
      device->sync();
    std::cout << "#bn.umesh: copying down world bounds"
              << std::endl;
    Device *anyDev = (*devices)[0];
    anyDev->rtc->copy(&worldBounds,
                      getPLD(anyDev)->pWorldBounds,
                      sizeof(worldBounds));
  }
  

  UMeshField::DD UMeshField::getDD(Device *device)
  {
    assert(device);
    UMeshField::DD dd;
    assert(vertices);
    assert(indices);
    assert(getPLD(device)->elements);
    dd.vertices = (float4 *)vertices->getDD(device);
    dd.indices  = (int    *)indices->getDD(device);
    dd.elements = (Element*)getPLD(device)->elements;
    dd.numElements = elementOffsets->count;
    assert(dd.vertices);
    assert(dd.indices);
    assert(dd.elements);
    return dd;
  }
  
  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
    auto sampler
      = std::make_shared<UMeshCUBQLSampler>(this);
    return std::make_shared<MCVolumeAccel<UMeshCUBQLSampler>>
      (volume,
       sampler,
       /* it's in mc/UMeshMC.dev.cu */
       "UMeshMC_ptx",
       "UMeshMC");
  }

}

RTC_DECLARE_COMPUTE(umeshRasterElements,barney::UMeshRasterElements);
RTC_DECLARE_COMPUTE(umeshCreateElements,barney::UMeshCreateElements);
RTC_DECLARE_COMPUTE(umeshComputeElementBBs,barney::UMeshComputeElementBBs);

