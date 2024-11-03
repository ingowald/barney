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
#include "barney/volume/MCGrid.cuh"
// just to be able to create these accelerators:
// #include "barney/umesh/mc/UMeshMCAccelerator.h"
#include "barney/umesh/mc/UMeshCUBQLSampler.h"
#include "barney/umesh/os/RTXObjectSpace.h"
#include "barney/umesh/os/AWT.h"

#define BUFFER_CREATE owlDeviceBufferCreate
// #define BUFFER_CREATE owlManagedMemoryBufferCreate

namespace barney {

  extern "C" char UMeshMC_ptx[];
  
  // this is probably waaaay overkill for smallish voluems, but those
  // are fast, anyway. and this helps for large ones...
  // enum { MC_GRID_SIZE = 256 };

  inline __device__ float length3(vec4f v)
  { return length(getPos(v)); }
  
  template<int D> inline __device__
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

#if 1
    vec4f oa,ob,oc,od0,od1;
    if (lab >= maxLen) {
      oa = ab;
      ob = c;
      oc = d;
      od0 = a;
      od1 = a;
      // rasterTet<D-1>(grid,ab,c,d,a);
      // rasterTet<D-1>(grid,ab,c,d,b);
    } else if (lac >= maxLen) {
      oa = ac;
      ob = b;
      oc = d;
      od0 = a;
      od1 = c;
      // rasterTet<D-1>(grid,ac,b,d,a);
      // rasterTet<D-1>(grid,ac,b,d,c);
    } else if (lad >= maxLen) {
      oa = ad;
      ob = b;
      oc = c;
      od0 = a;
      od1 = d;
      // rasterTet<D-1>(grid,ad,b,c,a);
      // rasterTet<D-1>(grid,ad,b,c,d);
    } else if (lbc >= maxLen) {
      oa = bc;
      ob = a;
      oc = d;
      od0 = b;
      od1 = c;
      // rasterTet<D-1>(grid,bc,a,d,b);
      // rasterTet<D-1>(grid,bc,a,d,c);
    } else if (lbd >= maxLen) {
      oa = bd;
      ob = a;
      oc = c;
      od0 = b;
      od1 = d;
      // rasterTet<D-1>(grid,bd,a,c,b);
      // rasterTet<D-1>(grid,bd,a,c,d);
    } else {
      oa = cd;
      ob = a;
      oc = b;
      od0 = c;
      od1 = d;
      // rasterTet<D-1>(grid,cd,a,b,c);
      // rasterTet<D-1>(grid,cd,a,b,d);
    }
    rasterTet<D-1>(grid,oa,ob,oc,od0);
    rasterTet<D-1>(grid,oa,ob,oc,od1);
#else
    if (lab >= maxLen) {
      rasterTet<D-1>(grid,ab,c,d,a);
      rasterTet<D-1>(grid,ab,c,d,b);
    } else if (lac >= maxLen) {
      rasterTet<D-1>(grid,ac,b,d,a);
      rasterTet<D-1>(grid,ac,b,d,c);
    } else if (lad >= maxLen) {
      rasterTet<D-1>(grid,ad,b,c,a);
      rasterTet<D-1>(grid,ad,b,c,d);
    } else if (lbc >= maxLen) {
      rasterTet<D-1>(grid,bc,a,d,b);
      rasterTet<D-1>(grid,bc,a,d,c);
    } else if (lbd >= maxLen) {
      rasterTet<D-1>(grid,bd,a,c,b);
      rasterTet<D-1>(grid,bd,a,c,d);
    } else {
      rasterTet<D-1>(grid,cd,a,b,c);
      rasterTet<D-1>(grid,cd,a,b,d);
    }
#endif
  }
  
  template<> inline __device__
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
  
  __global__ void rasterElements(MCGrid::DD grid,
                                 UMeshField::DD mesh)
  {
    const int eltIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (eltIdx >= mesh.numElements) return;    

    auto elt = mesh.elements[eltIdx];
    if (elt.type == Element::TET) {
      const vec4i indices = *(const vec4i *)&mesh.indices[elt.ofs0];
      vec4f a = make_vec4f(mesh.vertices[indices.x]);
      vec4f b = make_vec4f(mesh.vertices[indices.y]);
      vec4f c = make_vec4f(mesh.vertices[indices.z]);
      vec4f d = make_vec4f(mesh.vertices[indices.w]);
      rasterTet<5>(grid,a,b,c,d);
      return;
    }
//     if (elt.type == Element::GRID) {
//       int primID = elt.ID;

//       const box3f bounds = box3f((const vec3f &)mesh.gridDomains[primID].lower,
//                                  (const vec3f &)mesh.gridDomains[primID].upper);

//       vec3i numScalars = mesh.gridDims[primID]+1;
//       vec3f cellSize = bounds.size()/vec3f(mesh.gridDims[primID]);

//       const float *scalars = mesh.gridScalars + mesh.gridOffsets[primID];

//       auto linearIndex = [numScalars](const int x, const int y, const int z) {
//         return z*numScalars.y*numScalars.x + y*numScalars.x + x;
//       };

//       for (int z=0;z<mesh.gridDims[primID].z;z++) {
//         for (int y=0;y<mesh.gridDims[primID].y;y++) {
//           for (int x=0;x<mesh.gridDims[primID].x;x++) {
//             vec3i imin(x,y,z);
//             vec3i imax(x+1,y+1,z+1);

//             float f1 = scalars[linearIndex(imin.x,imin.y,imin.z)];
//             float f2 = scalars[linearIndex(imax.x,imin.y,imin.z)];
//             float f3 = scalars[linearIndex(imin.x,imax.y,imin.z)];
//             float f4 = scalars[linearIndex(imax.x,imax.y,imin.z)];

//             float f5 = scalars[linearIndex(imin.x,imin.y,imax.z)];
//             float f6 = scalars[linearIndex(imax.x,imin.y,imax.z)];
//             float f7 = scalars[linearIndex(imin.x,imax.y,imax.z)];
//             float f8 = scalars[linearIndex(imax.x,imax.y,imax.z)];

// #define EMPTY(x) isnan(x)
//             if (EMPTY(f1) || EMPTY(f2) || EMPTY(f3) || EMPTY(f4) ||
//                 EMPTY(f5) || EMPTY(f6) || EMPTY(f7) || EMPTY(f8))
//               continue;

//             float fmin = min(f1,min(f2,min(f3,min(f4,min(f5,min(f6,min(f7,f8)))))));
//             float fmax = max(f1,max(f2,max(f3,max(f4,max(f5,max(f6,max(f7,f8)))))));

//             const box4f cellBounds(vec4f(bounds.lower+vec3f(imin),fmin),
//                                    vec4f(bounds.lower+vec3f(imax),fmax));
//             rasterBox(grid,getBox(mesh.worldBounds),cellBounds);
//           }
//         }
//       }
//     } else
    {
      const box4f eltBounds = mesh.eltBounds(elt);
      rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
    }
  }

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
      assert(dev); assert(dev.get());
      SetActiveGPU forDuration(dev);
      auto d_mesh = getDD(dev);
      auto d_grid = grid.getDD(dev);
      rasterElements
        <<<divRoundUp(int(elements.size()),128),128>>>
        (d_grid,d_mesh);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }
    
  
  /*! computes - ON CURRENT DEVICE - the given mesh's prim bounds and
    per-prim scalar ranges, and writes those into givne
    pre-allocated device mem location */
  __global__
  void g_computeElementBoundingBoxes(box3f *d_primBounds,
                                     range1f *d_primRanges,
                                     UMeshField::DD mesh)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= mesh.numElements) return;

    auto elt = mesh.elements[tid];
    box4f eb = mesh.eltBounds(elt);
    d_primBounds[tid] = getBox(eb);
    if (d_primRanges) d_primRanges[tid] = getRange(eb);
  }

  /*! computes, on specified device, the bounding boxes and - if
    d_primRanges is non-null - the primitmives ranges. d_primBounds
    and d_primRanges (if non-null) must be pre-allocated and
    writeaable on specified device */
  void UMeshField::computeElementBBs(const Device::SP &device,
                                     box3f *d_primBounds,
                                     range1f *d_primRanges)
  {
    assert(device); assert(device.get());
    SetActiveGPU forDuration(device);
    int bs = 1024;
    int nb = divRoundUp(int(elements.size()),bs);
    g_computeElementBoundingBoxes
      <<<nb,bs>>>(d_primBounds,d_primRanges,getDD(device));
    BARNEY_CUDA_SYNC_CHECK();
  }

  UMeshField::UMeshField(Context *context, int slot,
                         std::vector<vec4f>   &_vertices,
                         std::vector<int>     &_indices,
                         std::vector<Element> &_elements,
                         const box3f &domain)
    : ScalarField(context,slot,domain),
      vertices(std::move(_vertices)),
      indices(std::move(_indices)),
      elements(std::move(_elements))
  {
    for (auto vtx : vertices) worldBounds.extend(getPos(vtx));

    if (!domain.empty())
      worldBounds = intersection(worldBounds,domain);

    assert(!elements.empty());

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
  }

  UMeshField::DD UMeshField::getDD(const Device::SP &device)
  {
    assert(device.get());
    UMeshField::DD dd;
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
    
    return dd;
  }
  

  ScalarField::SP UMeshField::create(Context *context, int slot,
                                     const vec4f   *_vertices, int numVertices,
                                     const int     *_indices,  int numIndices,
                                     const int     *elementOffsets,
                                     int      numElements,
                                     const box3f &domain)
  {
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
  }
  
  void UMeshField::setVariables(OWLGeom geom)
  {
    ScalarField::setVariables(geom);
    
    owlGeomSetBuffer(geom,"umesh.vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"umesh.indices",indicesBuffer);
    owlGeomSetBuffer(geom,"umesh.elements",elementsBuffer);
  }
  
  void UMeshField::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    ScalarField::DD::addVars(vars,base);
    std::vector<OWLVarDecl> mine = 
      {
        { "umesh.vertices",    OWL_BUFPTR, base+OWL_OFFSETOF(DD,vertices) },
        { "umesh.indices" ,    OWL_BUFPTR, base+OWL_OFFSETOF(DD,indices) },
        { "umesh.elements",    OWL_BUFPTR, base+OWL_OFFSETOF(DD,elements) },
      };
    for (auto var : mine)
      vars.push_back(var);
  }

  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
#if 1
    const char *methodFromEnv = getenv("BARNEY_UMESH");
    std::string method = (methodFromEnv ? methodFromEnv : "DDA");

    if (method == "DDA" || method == "MCDDA" || method == "dda") {
      return std::make_shared<MCDDAVolumeAccel<UMeshCUBQLSampler>::Host>
        (this,volume,UMeshMC_ptx);
    }

    if (method == "MCRTX")
      return std::make_shared<MCRTXVolumeAccel<UMeshCUBQLSampler>::Host>
        (this,volume,UMeshMC_ptx);
    
    if (method == "OS" || method == "os")
      return std::make_shared<RTXObjectSpace::Host>
        (this,volume);
    
    if (method == "AWT" || method == "awt")
      return std::make_shared<UMeshAWT::Host>
        (this,volume);
    
    throw std::runtime_error("unknown BARNEY_UMESH accelerator method");
#else
    const char *methodFromEnv = getenv("BARNEY_UMESH");
    std::string method = (methodFromEnv ? methodFromEnv : "object-space");
    if (method == "macro-cells" || method == "spatial" || method == "mc")
      return std::make_shared<UMeshAccel_MC_CUBQL>(this,volume);
    else if (method == "AWT" || method == "awt")
      return std::make_shared<UMeshAWT>(this,volume);
    else if (method == "object-space" || method == "os")
      return std::make_shared<RTXObjectSpace>(this,volume);
    else
      throw std::runtime_error("found BARNEY_METHOD env-var, but didn't recognize its value. allowed values are 'awt', 'object-space', and 'macro-cells'");
#endif
  }


}
  
