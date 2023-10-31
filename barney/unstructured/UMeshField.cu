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

#include "barney/unstructured/UMeshField.h"
#include "barney/Context.h"
#include "barney/unstructured/UMeshMCAccelerator.h"
#include "barney/unstructured/UMeshRTXObjectSpace.h"

namespace barney {

  enum { MC_GRID_SIZE = 32 };
  
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
  
  __global__ void clearMCs(MCGrid::DD grid)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= grid.dims.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= grid.dims.y) return;
    int iz = threadIdx.z+blockIdx.z*blockDim.z; if (iz >= grid.dims.z) return;
    
    int ii = ix + grid.dims.x*(iy + grid.dims.y*(iz));
    grid.scalarRanges[ii] = { 1e+30f,-1e+30f };//range1f();
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

    const vec3i bs = 4;
    const vec3i nb = divRoundUp(dims,bs);
    for (auto dev : devGroup->devices) {
      SetActiveGPU forDuration(dev);
      BARNEY_CUDA_SYNC_CHECK();
      std::cout << "clearing macro cells" << std::endl;
      auto d_grid = grid.getDD(dev->owlID);
      clearMCs
        <<<(dim3)nb,(dim3)bs>>>
        (d_grid);
      BARNEY_CUDA_SYNC_CHECK();

      auto d_mesh = getDD(dev->owlID);
      rasterElements
        <<<divRoundUp(int(elements.size()),1024),1024>>>
        (d_grid,d_mesh);
      BARNEY_CUDA_SYNC_CHECK();
    }
  }
    
  
  __global__
  void computeElementBoundingBoxes(box3f *d_primBounds, UMeshField::DD mesh)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= mesh.numElements) return;

    auto elt = mesh.elements[tid];
    d_primBounds[tid] = getBox(mesh.eltBounds(elt));
  }


  UMeshField::UMeshField(DevGroup *devGroup,
                         std::vector<vec4f> &_vertices,
                         std::vector<TetIndices> &_tetIndices,
                         std::vector<PyrIndices> &_pyrIndices,
                         std::vector<WedIndices> &_wedIndices,
                         std::vector<HexIndices> &_hexIndices)
    : ScalarField(devGroup),
      vertices(std::move(_vertices)),
      tetIndices(std::move(_tetIndices)),
      pyrIndices(std::move(_pyrIndices)),
      wedIndices(std::move(_wedIndices)),
      hexIndices(std::move(_hexIndices))
  {
    for (auto vtx : vertices) worldBounds.extend(vtx);
    for (int i=0;i<tetIndices.size();i++)
      elements.push_back(Element(i,Element::TET));

    assert(!elements.empty());
    
    for (int i=0;i<hexIndices.size();i++)
      elements.push_back(Element(i,Element::HEX));
    
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
  }

  UMeshField::DD UMeshField::getDD(int devID)
  {
    UMeshField::DD dd;

    dd.vertices    = (const float4  *)owlBufferGetPointer(verticesBuffer,devID);
    dd.tetIndices  = (const int4    *)owlBufferGetPointer(tetIndicesBuffer,devID);
    dd.hexIndices  = (const ints<8> *)owlBufferGetPointer(hexIndicesBuffer,devID);
    dd.elements    = (const Element *)owlBufferGetPointer(elementsBuffer,devID);
    dd.numElements = elements.size();
    dd.worldBounds = worldBounds;

    return dd;
  }

  
  ScalarField *DataGroup::createUMesh(std::vector<vec4f> &vertices,
                                      std::vector<TetIndices> &tetIndices,
                                      std::vector<PyrIndices> &pyrIndices,
                                      std::vector<WedIndices> &wedIndices,
                                      std::vector<HexIndices> &hexIndices)
  {
    ScalarField::SP sf
      = std::make_shared<UMeshField>(devGroup.get(),
                                     vertices,
                                     tetIndices,
                                     pyrIndices,
                                     wedIndices,
                                     hexIndices);
    return getContext()->initReference(sf);
  }
  
  VolumeAccel::SP UMeshField::createAccel(Volume *volume)
  {
#if 0
    return std::make_shared<UMeshAccel_MC_CUBQL>(this,volume);
    // return std::make_shared<UMeshAccel_MC_CUBQL>(this,volume);
#else
    return std::make_shared<UMeshRTXObjectSpace>(this,volume);
#endif
  }

}
  
