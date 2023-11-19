// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/umesh/UMeshField.h"
#include "barney/Context.h"
// just to be able to create these accelerators:
#include "barney/umesh/UMeshMCAccelerator.h"
#include "barney/umesh/RTXObjectSpace.h"

namespace barney {

  enum { MC_GRID_SIZE = 128 };
  
  inline __device__
  float fatomicMin(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old <= value) return old;
    do {
      assumed = old;
      old = atomicCAS((unsigned int*)addr,
                      __float_as_int(assumed),
                      __float_as_int(value));
    } while(old!=assumed);
    return old;
  }

  inline __device__
  float fatomicMax(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old >= value) return old;
    do {
      assumed = old;
      old = atomicCAS((unsigned int*)addr,
                      __float_as_int(assumed),
                      __float_as_int(value));
    } while(old!=assumed);
    return old;
  }
  
  inline __device__
  int project(float f,
              const interval<float> range,
              int dim)
  {
    return max(0,min(dim-1,int(dim*(f-range.lower)/(range.upper-range.lower))));
  }

  inline __device__
  vec3i project(const vec3f &pos,
                const box3f &bounds,
                const vec3i &dims)
  {
    return vec3i(project(pos.x,{bounds.lower.x,bounds.upper.x},dims.x),
                 project(pos.y,{bounds.lower.y,bounds.upper.y},dims.y),
                 project(pos.z,{bounds.lower.z,bounds.upper.z},dims.z));
  }

  inline __device__
  void rasterBox(MCGrid::DD grid,
                 const box3f worldBounds,
                 const box4f primBounds4,
                 bool dbg=false)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = project(pb.lower,worldBounds,grid.dims);
    vec3i hi = project(pb.upper,worldBounds,grid.dims);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const int cellID
            = ix
            + iy * grid.dims.x
            + iz * grid.dims.x * grid.dims.y;
          auto &cell = grid.scalarRanges[cellID];
          fatomicMin(&cell.lower,primBounds4.lower.w);
          fatomicMax(&cell.upper,primBounds4.upper.w);
        }
  }
  
  __global__ void rasterElements(MCGrid::DD grid,
                                 UMeshField::DD mesh)
  {
    const int eltIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (eltIdx >= mesh.numElements) return;    

    auto elt = mesh.elements[eltIdx];
    const box4f eltBounds = mesh.eltBounds(elt);
    rasterBox(grid,getBox(mesh.worldBounds),eltBounds);
  }

  /*! build *initial* macro-cell grid (ie, the scalar field min/max
    ranges, but not yet the majorants) over a umesh */
  void UMeshField::buildInitialMacroCells(MCGrid &grid)
  {
    if (grid.built()) {
      // initial grid already built
      return;
    }
    
    std::cout << OWL_TERMINAL_BLUE
              << "#bn.um: building initial macro cell grid"
              << OWL_TERMINAL_DEFAULT << std::endl;

    PRINT(worldBounds);
    float maxWidth = reduce_max(getBox(worldBounds).size());
    vec3i dims = 1+vec3i(getBox(worldBounds).size() * ((MC_GRID_SIZE-1) / maxWidth));
    printf("#bn.um: chosen macro-cell dims of (%i %i %i)\n",
           dims.x,
           dims.y,
           dims.z);
    std::cout << "allcating macro cells" << std::endl;
    grid.resize(dims);

    std::cout << "clearing macro cells" << std::endl;
    grid.clearCells();
    
    std::cout << "building macro cells" << std::endl;
    const vec3i bs = 4;
    const vec3i nb = divRoundUp(dims,bs);
    for (auto dev : devGroup->devices) {
      SetActiveGPU forDuration(dev);
      auto d_mesh = getDD(dev->owlID);
      auto d_grid = grid.getDD(dev->owlID);
      rasterElements
        <<<divRoundUp(int(elements.size()),1024),1024>>>
        (d_grid,d_mesh);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }
    
  
  /*! computes - ON CURRENT DEVICE - the given mesh's prim bounds, and
      writes those into givne pre-allocated device mem location */
  __global__

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
  void UMeshField::computeElementBBs(int deviceID,
                                     box3f *d_primBounds,
                                     range1f *d_primRanges)
  {
    SetActiveGPU forDuration(devGroup->devices[deviceID]);
    int bs = 1024;
    int nb = divRoundUp(int(elements.size()),bs);
    g_computeElementBoundingBoxes
      <<<nb,bs>>>(d_primBounds,d_primRanges,getDD(deviceID));
    BARNEY_CUDA_SYNC_CHECK();
  }

  UMeshField::UMeshField(DevGroup *devGroup,
                         std::vector<vec4f> &_vertices,
                         std::vector<TetIndices> &_tetIndices,
                         std::vector<PyrIndices> &_pyrIndices,
                         std::vector<WedIndices> &_wedIndices,
                         std::vector<HexIndices> &_hexIndices,
                         std::vector<int> &_gridOffsets,
                         std::vector<vec3i> &_gridDims,
                         std::vector<box4f> &_gridDomains,
                         std::vector<float> &_gridScalars)
    : ScalarField(devGroup),
      vertices(std::move(_vertices)),
      tetIndices(std::move(_tetIndices)),
      pyrIndices(std::move(_pyrIndices)),
      wedIndices(std::move(_wedIndices)),
      hexIndices(std::move(_hexIndices)),
      gridOffsets(std::move(_gridOffsets)),
      gridDims(std::move(_gridDims)),
      gridDomains(std::move(_gridDomains)),
      gridScalars(std::move(_gridScalars))
  {
    for (auto vtx : vertices) worldBounds.extend(vtx);
    for (auto dom : gridDomains) worldBounds.extend(dom);
    for (int i=0;i<tetIndices.size();i++)
      elements.push_back(Element(i,Element::TET));

    for (int i=0;i<pyrIndices.size();i++)
      elements.push_back(Element(i,Element::PYR));

    for (int i=0;i<wedIndices.size();i++)
      elements.push_back(Element(i,Element::WED));

    for (int i=0;i<hexIndices.size();i++)
      elements.push_back(Element(i,Element::HEX));
    
    for (int i=0;i<gridOffsets.size();i++)
      elements.push_back(Element(i,Element::GRID));
    
    assert(!elements.empty());
    
    verticesBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_FLOAT4,
                              vertices.size(),
                              vertices.data());
    tetIndicesBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              4*tetIndices.size(),
                              tetIndices.data());

    pyrIndicesBuffer
        = owlDeviceBufferCreate(getOWL(),
                                OWL_INT,
                                5*pyrIndices.size(),
                                pyrIndices.data());
    wedIndicesBuffer
        = owlDeviceBufferCreate(getOWL(),
                                OWL_INT,
                                6*wedIndices.size(),
                                wedIndices.data());
    
    assert(sizeof(ints<8>) == 8*sizeof(int));
    hexIndicesBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              8*hexIndices.size(),
                              hexIndices.data());

    assert(sizeof(Element) == sizeof(int));
    elementsBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              elements.size(),
                              elements.data());

    gridOffsetsBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT,
                              gridOffsets.size(),
                              gridOffsets.data());
    gridDimsBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_INT3,
                              gridDims.size(),
                              gridDims.data());
    gridDomainsBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_USER_TYPE(box4f),
                              gridDomains.size(),
                              gridDomains.data());

    gridScalarsBuffer
      = owlDeviceBufferCreate(getOWL(),
                              OWL_FLOAT,
                              gridScalars.size(),
                              gridScalars.data());
  }

  UMeshField::DD UMeshField::getDD(int devID)
  {
    UMeshField::DD dd;

    dd.vertices    = (const float4  *)owlBufferGetPointer(verticesBuffer,devID);
    dd.tetIndices  = (const int4    *)owlBufferGetPointer(tetIndicesBuffer,devID);
    dd.pyrIndices  = (const ints<5> *)owlBufferGetPointer(pyrIndicesBuffer,devID);
    dd.wedIndices  = (const ints<6> *)owlBufferGetPointer(wedIndicesBuffer,devID);
    dd.hexIndices  = (const ints<8> *)owlBufferGetPointer(hexIndicesBuffer,devID);
    dd.elements    = (const Element *)owlBufferGetPointer(elementsBuffer,devID);
    dd.gridOffsets = (const int     *)owlBufferGetPointer(gridOffsetsBuffer,devID);
    dd.gridDims    = (const vec3i   *)owlBufferGetPointer(gridDimsBuffer,devID);
    dd.gridScalars = (const float   *)owlBufferGetPointer(gridScalarsBuffer,devID);
    dd.gridDomains = (const box4f   *)owlBufferGetPointer(gridDomainsBuffer,devID);
    dd.gridScalars = (const float   *)owlBufferGetPointer(gridScalarsBuffer,devID);
    dd.numElements = elements.size();
    dd.worldBounds = worldBounds;

    return dd;
  }

  
  ScalarField *DataGroup::createUMesh(std::vector<vec4f> &vertices,
                                      std::vector<TetIndices> &tetIndices,
                                      std::vector<PyrIndices> &pyrIndices,
                                      std::vector<WedIndices> &wedIndices,
                                      std::vector<HexIndices> &hexIndices,
                                      std::vector<int> &gridOffsets,
                                      std::vector<vec3i> &gridDims,
                                      std::vector<box4f> &gridDomains,
                                      std::vector<float> &gridScalars)
  {
    ScalarField::SP sf
      = std::make_shared<UMeshField>(devGroup.get(),
                                     vertices,
                                     tetIndices,
                                     pyrIndices,
                                     wedIndices,
                                     hexIndices,
                                     gridOffsets,
                                     gridDims,
                                     gridDomains,
                                     gridScalars);
    return getContext()->initReference(sf);
  }
  
  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
    const char *methodFromEnv = getenv("BARNEY_METHOD");
    std::string method = (methodFromEnv ? methodFromEnv : "");
    if (method == "" || method == "macro-cells" || method == "spatial")
      return std::make_shared<UMeshAccel_MC_CUBQL>(this,volume);
    else if (method == "AWT" || method == "awt")
      return std::make_shared<UMeshAWT>(this,volume);
    else if (method == "object-space")
      return std::make_shared<RTXObjectSpace>(this,volume);
    else throw std::runtime_error("found BARNEY_METHOD env-var, but didn't recognize its value. allowed values are 'awt', 'object-space', and 'macro-cells'");
  }

  void UMeshField::setVariables(OWLGeom geom, bool firstTime)
  {
    ScalarField::setVariables(geom,firstTime);
    
    // ------------------------------------------------------------------
    assert(mesh->tetIndicesBuffer);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
      
    owlGeomSetBuffer(geom,"tetIndices",tetIndicesBuffer);
    owlGeomSetBuffer(geom,"pyrIndices",pyrIndicesBuffer);
    owlGeomSetBuffer(geom,"wedIndices",wedIndicesBuffer);
    owlGeomSetBuffer(geom,"hexIndices",hexIndicesBuffer);
    owlGeomSetBuffer(geom,"elements",elementsBuffer);
    owlGeomSetBuffer(geom,"gridOffsets",gridOffsetsBuffer);
    owlGeomSetBuffer(geom,"gridDims",gridDimsBuffer);
    owlGeomSetBuffer(geom,"gridDomains",gridDomainsBuffer);
    owlGeomSetBuffer(geom,"gridScalars",gridScalarsBuffer);
  }
  
  std::vector<OWLVarDecl> UMeshField::getVarDecls(uint32_t myOfs)
  {
    std::vector<OWLVarDecl> mine = 
      {
       { "mesh.vertices",    OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,vertices) },
       { "mesh.tetIndices",  OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,tetIndices) },
       { "mesh.pyrIndices",  OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,pyrIndices) },
       { "mesh.wedIndices",  OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,wedIndices) },
       { "mesh.hexIndices",  OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,hexIndices) },
       { "mesh.elements",    OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,elements) },
       { "mesh.gridOffsets", OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,gridOffsets) },
       { "mesh.gridDims",    OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,gridDims) },
       { "mesh.gridDomains", OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,gridDomains) },
       { "mesh.gridScalars", OWL_BUFPTR, myOfs+OWL_OFFSETOF(DD,gridScalars) },
      };
    for (auto var : ScalarField::getVarDecls(myOfs))
      mine.push_back(var);
    return mine;
  };
}
  
