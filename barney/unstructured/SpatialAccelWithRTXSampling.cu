// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
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

/*! dimensions of the macro cell grid, in widest dimension */
#define MC_GRID_SIZE 64

namespace vopat {

#if 0
#if UMESH_SHARED_FACES
#else
#endif
  
  void UMeshVolume::build(OWLContext owl,
                          OWLModule owlDevCode)
  {
    OWLVarDecl args[] = {
                         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(Geom,vertices) },
                         { "scalars", OWL_BUFPTR, OWL_OFFSETOF(Geom,scalars) },
                         { "tets", OWL_BUFPTR, OWL_OFFSETOF(Geom,tets) },
                         { "numTets", OWL_INT, OWL_OFFSETOF(Geom,numTets) },
#if UMESH_SHARED_FACES
                         { "tetsOnFace", OWL_BUFPTR, OWL_OFFSETOF(Geom,tetsOnFace) },
#else
#endif
                         { nullptr }
    };
    gt = owlGeomTypeCreate(owl,
#if UMESH_SHARED_FACES
                           OWL_TRIANGLES,
#else
                           OWL_GEOMETRY_USER,
#endif
                           sizeof(Geom),
                           args,-1);
#if UMESH_SHARED_FACES
    owlGeomTypeSetClosestHit(gt,0,owlDevCode,"UMeshGeomCH");
#else
    owlGeomTypeSetBoundsProg(gt,owlDevCode,"UMeshGeomBounds");
    owlGeomTypeSetIntersectProg(gt,0,owlDevCode,"UMeshGeomIS");
    owlGeomTypeSetAnyHit(gt,0,owlDevCode,"UMeshGeomAH");
    owlGeomTypeSetClosestHit(gt,0,owlDevCode,"UMeshGeomCH");
#endif

    scalarsBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT,
                                          myBrick->umesh->perVertex->values.size(),
                                          myBrick->umesh->perVertex->values.data());
    
    verticesBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT3,
                                           myBrick->umesh->vertices.size(),
                                           myBrick->umesh->vertices.data());
    
    tetsBuffer = owlDeviceBufferCreate(owl,OWL_INT4,
                                       myBrick->umesh->tets.size(),
                                       myBrick->umesh->tets.data());

#if UMESH_SHARED_FACES
    CUDA_SYNC_CHECK();
    std::cout << "building shared faces accel" << std::endl;
    std::map<vec3i,int> faceID;
    std::vector<vec3i> sharedFaceIndices;
    std::vector<vec2i> sharedFaceNeighbors;
    for (auto tet : myBrick->umesh->tets)
      iterateFaces(tet,
                   [&faceID,&sharedFaceIndices,&sharedFaceNeighbors]
                   (const vec3i faceVertices, int side)
                   {
                     if (faceID.find(faceVertices) == faceID.end()) {
                       faceID[faceVertices] = sharedFaceIndices.size();
                       sharedFaceIndices.push_back(faceVertices);
                       sharedFaceNeighbors.push_back(vec2i(-1));
                     }
                   });
    
    for (int tetID=0;tetID<myBrick->umesh->tets.size();tetID++) {
      auto tet = myBrick->umesh->tets[tetID];
      iterateFaces(myBrick->umesh->tets[tetID],[&](vec3i faceVertices, int side){
          sharedFaceNeighbors[faceID[faceVertices]][side] = tetID;
        });
    }

    std::cout << "shared faces stuff built on host, setting up device geom" << std::endl;
    CUDA_SYNC_CHECK();
    sharedFaceIndicesBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT3,
                              sharedFaceIndices.size(),sharedFaceIndices.data());
    sharedFaceNeighborsBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT2,
                              sharedFaceNeighbors.size(),sharedFaceNeighbors.data());
#else
#endif
    geom = owlGeomCreate(owl,gt);
    
    owlGeomSet1i(geom,"numTets",int(myBrick->umesh->tets.size()));
    owlGeomSetBuffer(geom,"tets",tetsBuffer);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"scalars",scalarsBuffer);

#if UMESH_SHARED_FACES
    owlTrianglesSetVertices(geom,verticesBuffer,
                            myBrick->umesh->vertices.size(),sizeof(vec3f),0);
    owlTrianglesSetIndices(geom,sharedFaceIndicesBuffer,
                           sharedFaceIndices.size(),sizeof(vec3i),0);
    owlGeomSetBuffer(geom,"tetsOnFace",sharedFaceNeighborsBuffer);
    blas = owlTrianglesGeomGroupCreate(owl,1,&geom);
#else
    owlGeomSetPrimCount(geom,myBrick->umesh->tets.size());
    owlBuildPrograms(owl);
    blas = owlUserGeomGroupCreate(owl,1,&geom);
#endif

    // and put this into a single-instance tlas
    owlGroupBuildAccel(blas);
    tlas = owlInstanceGroupCreate(owl,1,&blas);
    
    owlGroupBuildAccel(tlas);
    globals.sampleAccel = owlGroupGetTraversable(tlas,0);
    globals.domain = myBrick->domain;
    
    CUDA_SYNC_CHECK();
    std::cout << "shared faces stuff built. done" << std::endl;
  }
  
  void UMeshVolume::setDD(OWLLaunchParams lp) 
  {
    owlParamsSetRaw(lp,"volumeSampler.umesh",&globals);
    owlParamsSet1i(lp,"volumeSampler.type",int(VolumeSamplerType_UMesh));
  }
  
  void UMeshVolume::addLPVars(std::vector<OWLVarDecl> &lpVars) 
  {
    lpVars.push_back({"volumeSampler.umesh",OWL_USER_TYPE(DD),
                      OWL_OFFSETOF(LaunchParams,volumeSampler.umesh)});
  }
  






  inline __device__
  float fatomicMin(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old <= value) return old;
    do {
      assumed = old;
      old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
        
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
      old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
        
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
  void rasterBox(MacroCell *d_mcGrid,
                 const vec3i dims,
                 const box3f worldBounds,
                 const box4f primBounds4,
                 bool dbg=false)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = project(pb.lower,worldBounds,dims);
    vec3i hi = project(pb.upper,worldBounds,dims);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const int cellID
            = ix
            + iy * dims.x
            + iz * dims.x * dims.y;
          auto &cell = d_mcGrid[cellID].inputRange;
          fatomicMin(&cell.lower,primBounds4.lower.w);
          fatomicMax(&cell.upper,primBounds4.upper.w);
        }
  }
  
  __global__ void clearMCs(MacroCell *mcData,
                           vec3i dims)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= dims.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= dims.y) return;
    int iz = threadIdx.z+blockIdx.z*blockDim.z; if (iz >= dims.z) return;

    int ii = ix + dims.x*(iy + dims.y*(iz));
    mcData[ii].inputRange.lower = +FLT_MAX;
    mcData[ii].inputRange.upper = -FLT_MAX;
  }
  
  __global__ void rasterTets(MacroCell *mcData,
                             vec3i mcDims,
                             box3f domain,
                             vec3f *vertices,
                             float *scalars,
                             umesh::UMesh::Tet *tets,
                             int numTets)
  {
    const int blockID
      = blockIdx.x
      + blockIdx.y * 1024
      ;
    const int primIdx = blockID*blockDim.x + threadIdx.x;
    if (primIdx >= numTets) return;    

    umesh::UMesh::Tet tet = tets[primIdx];
    const box4f primBounds4 = box4f()
      .including(vec4f(vertices[tet.x],scalars[tet.x]))
      .including(vec4f(vertices[tet.y],scalars[tet.y]))
      .including(vec4f(vertices[tet.z],scalars[tet.z]))
      .including(vec4f(vertices[tet.w],scalars[tet.w]));

    rasterBox(mcData,mcDims,domain,primBounds4);
  }
  
  /*! build the given macro cell grid uses this volume's data */
  void UMeshVolume::buildMCs(MCGrid &mcGrid) 
  {
    std::cout << OWL_TERMINAL_BLUE
              << "#vopat.umesh: building macro cells .."
              << OWL_TERMINAL_DEFAULT
              << std::endl;

    float maxWidth = reduce_max(myBrick->domain.size());
    mcGrid.dd.dims = 1+vec3i(myBrick->domain.size() * ((MC_GRID_SIZE-1) / maxWidth));
    printf("#vopat.umesh(%i): chosen macro-cell dims of (%i %i %i)\n",
           myBrick->ID,
           mcGrid.dd.dims.x,
           mcGrid.dd.dims.y,
           mcGrid.dd.dims.z);

    double t0 = getCurrentTime();
    mcGrid.cells.resize(volume(mcGrid.dd.dims));
    mcGrid.dd.cells = mcGrid.cells.get();
    CUDA_SYNC_CHECK();
    clearMCs<<<(dim3)divRoundUp(mcGrid.dd.dims,vec3i(4)),(dim3)vec3i(4)>>>
      (mcGrid.cells.get(),mcGrid.dd.dims);
    CUDA_SYNC_CHECK();
    mcGrid.dd.cells  = mcGrid.cells.get();

    if (myBrick->umesh->tets.empty())
      throw std::runtime_error("no tets!?");
    const unsigned int blockSize = 128;
    unsigned int numTets = (int)myBrick->umesh->tets.size();
    const unsigned int numBlocks = divRoundUp(numTets,blockSize*1024u);
    dim3 _nb{1024u,numBlocks,1u};
    dim3 _bs{blockSize,1u,1u};
    rasterTets<<<_nb,_bs>>>
      (mcGrid.dd.cells,
       mcGrid.dd.dims,
       myBrick->domain,
       (vec3f*)owlBufferGetPointer(verticesBuffer,0),
       (float*)owlBufferGetPointer(scalarsBuffer,0),
       (umesh::UMesh::Tet*)owlBufferGetPointer(tetsBuffer,0),
       myBrick->umesh->tets.size());
    CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << OWL_TERMINAL_GREEN
              << "#vopat.umesh: done building macro cells; took " << prettyDouble(t1-t0) << "s..."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    affine3f toMcGrid
      = linear3f::scale(vec3f(mcGrid.dd.dims));
    affine3f toWorldDomain
      = { linear3f::scale(myBrick->domain.size()), myBrick->domain.lower };
    mcGrid.dd.worldToMcSpace
      = toMcGrid * rcp(toWorldDomain);
  }
#endif
}
