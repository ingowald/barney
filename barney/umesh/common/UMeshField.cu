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

  // this is probably waaaay overkill for smallish voluems, but those
  // are fast, anyway. and this helps for large ones...
  // enum { MC_GRID_SIZE = 256 };

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
  
  struct RasterElements {
    /* kernel data */
    MCGrid::DD     grid;
    UMeshField::DD mesh;
    
    template<typename ComputeInterface>
    inline __both__
    void rasterElements(ComputeInterface &ti)
    {
      const int eltIdx = ti.getBlockIdx().x*ti.getBlockDim().x + ti.getThreadIdx().x;
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
  
  void UMeshField::buildMCs(MCGrid &grid)
  {
    buildInitialMacroCells(grid);
  }
  
  /*! build *initial* macro-cell grid (ie, the scalar field min/max
    ranges, but not yet the majorants) over a umesh */
  void UMeshField::buildInitialMacroCells(MCGrid &grid)
  {
    BARNEY_NYI();
#if 0
    if (grid.built()) {
      // initial grid already built
      return;
    }
    
    float maxWidth = reduce_max(getBox(worldBounds).size());
    int MC_GRID_SIZE
      = 128 + int(sqrtf((float)elements.size())/30);
    vec3i dims = 1+vec3i(getBox(worldBounds).size() * ((MC_GRID_SIZE-1) / maxWidth));
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.um: building initial macro cell grid of " << dims << " MCs"
              << OWL_TERMINAL_DEFAULT << std::endl;
    grid.resize(dims);

    grid.gridOrigin
      = worldBounds.lower;
    grid.gridSpacing
      = worldBounds.size() * rcp(vec3f(dims));
    
    grid.clearCells();
    
    const vec3i bs = 4;
    const vec3i nb = divRoundUp(dims,bs);
    for (auto dev : getDevices()) {
      SetActiveGPU forDuration(dev);
      auto d_mesh = getDD(dev);
      auto d_grid = grid.getDD(dev);
      CHECK_CUDA_LAUNCH(rasterElements,
                        divRoundUp(int(elements.size()),128),128,0,0,
                        //
                        d_grid,d_mesh);
      // rasterElements
      // <<<divRoundUp(int(elements.size()),128),128>>>
      // (d_grid,d_mesh);
      BARNEY_CUDA_SYNC_CHECK();
    }
#endif
  }
    
  
  /*! computes - ON CURRENT DEVICE - the given mesh's prim bounds and
    per-prim scalar ranges, and writes those into givne
    pre-allocated device mem location */
  struct ComputeElementBoundingBoxes {
    /* kernel ARGS */
    box3f         *d_primBounds;
    range1f       *d_primRanges;
    UMeshField::DD mesh;
    
    /* kernel FUNCTION */
    template<typename ComputeInterface>
    inline __both__
    void run(ComputeInterface &ti)
    {
      const int tid = ti.getBlockIdx().x*ti.getBlockDim().x + ti.getThreadIdx().x;
      if (tid >= mesh.numElements) return;
      
      auto elt = mesh.elements[tid];
      box4f eb = mesh.eltBounds(elt);
      d_primBounds[tid] = getBox(eb);
      if (d_primRanges) d_primRanges[tid] = getRange(eb);
    }
  };

  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitmives ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeaable on specified device */
  void UMeshField::computeElementBBs(Device  *device,
                                     box3f   *d_primBounds,
                                     range1f *d_primRanges)
  {
#if 1
    BARNEY_NYI();
#else
    assert(device);
    SetActiveGPU forDuration(device);
    int bs = 1024;
    int nb = divRoundUp(int(elements.size()),bs);
    CHECK_CUDA_LAUNCH(g_computeElementBoundingBoxes,
                      nb,bs,0,0,
                      d_primBounds,d_primRanges,getDD(device));
    // g_computeElementBoundingBoxes
    //   <<<nb,bs>>>(d_primBounds,d_primRanges,getDD(device));
    BARNEY_CUDA_SYNC_CHECK();
#endif
  }

  UMeshField::UMeshField(Context *context, 
                         const DevGroup::SP   &devices,
                         std::vector<vec4f>   &_vertices,
                         std::vector<int>     &_indices,
                         std::vector<Element> &_elements,
                         const box3f &domain)
    : ScalarField(context,devices),
      vertices(std::move(_vertices)),
      indices(std::move(_indices)),
      elements(std::move(_elements))
  {
    for (auto vtx : vertices) worldBounds.extend(getPos(vtx));

    if (!domain.empty())
      worldBounds = intersection(worldBounds,domain);

    assert(!elements.empty());

    BARNEY_NYI();
#if 0
    verticesBuffer
      = BUFFER_CREATE(getOWL(),
                      OWL_FLOAT4,
                      vertices.size(),
                      vertices.data());
    indicesBuffer
      = BUFFER_CREATE(getOWL(),
                      OWL_INT,
                      indices.size(),
                      indices.data());

    elementsBuffer
      = BUFFER_CREATE(getOWL(),
                      OWL_INT,
                      elements.size(),
                      elements.data());
#endif
  }

  UMeshField::DD UMeshField::getDD(Device *device)
  {
    assert(device);
    UMeshField::DD dd;
    BARNEY_NYI();
#if 0
    int devID = device->owlID;
    assert(verticesBuffer);
    assert(indicesBuffer);
    assert(elementsBuffer);
    dd.vertices
      = (const float4  *)owlBufferGetPointer(verticesBuffer,devID);
    dd.indices
      = (const int     *)owlBufferGetPointer(indicesBuffer,devID);
    dd.elements
      = (const Element *)owlBufferGetPointer(elementsBuffer,devID);
    dd.numElements
      = (int)elements.size();
    dd.worldBounds
      = worldBounds;
#endif    
    return dd;
  }
  
  
  UMeshField::SP UMeshField::create(Context            *context,
                                    const DevGroup::SP &devices,
                                    const vec4f        *_vertices,
                                    int                 numVertices,
                                    const int          *_indices,
                                    int                 numIndices,
                                     const int          *elementOffsets,
                                     int                 numElements,
                                     const box3f        &domain)
  {
    BARNEY_NYI();
#if 0
    std::vector<Element> elements;
    for (int i=0;i<numElements;i++) {
      Element elt;
      elt.ofs0 = elementOffsets[i];
      if (elt.ofs0 != elementOffsets[i])
        throw std::runtime_error("not enough bits to encode element offset");

      int eltEnd
        = (i==(numElements-1))
        ? numIndices
        : elementOffsets[i+1];
      int numEltIndices = eltEnd - elt.ofs0;

      switch (numEltIndices) {
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
        throw std::runtime_error("non-supported element type with "
                                 +std::to_string(numEltIndices)+" indices");
      }
      elements.push_back(elt);
    }
    std::vector<vec4f> vertices(numVertices);
    std::copy(_vertices,_vertices+numVertices,vertices.data());
    std::vector<int> indices(numIndices);
    std::copy(_indices,_indices+numIndices,indices.data());
    ScalarField::SP sf
      = std::make_shared<UMeshField>(context,slot,
                                     vertices,
                                     indices,
                                     elements,
                                     domain);
    return sf;
#endif
  }
  
  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
    return std::make_shared<MCVolumeAccel<UMeshCUBQLSampler>>(this,volume);
  }

}
  
