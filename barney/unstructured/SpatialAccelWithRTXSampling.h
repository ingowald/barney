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

#pragma once

#include "vopat/volume/Volume.h"
#include "model/UMeshModel.h"

namespace vopat {

  /*! if set to true, we'll use triangle accel for shared-face tets;
      if not, we'll use user geom with actual point-in-tet test in
      isec program (probably slower, but should need less memory) */
  // #define UMESH_SHARED_FACES 1
  
  struct UMeshVolume : public Volume {
    typedef std::shared_ptr<UMeshVolume> SP;

    static SP create(UMeshBrick::SP brick)
    { return std::make_shared<UMeshVolume>(brick); }
    
    UMeshVolume(UMeshBrick::SP brick)
      : Volume(brick), myBrick(brick), umesh(brick->umesh)
    {}
    
    struct SamplePRD {
      float sampledValue;
    };

    struct Geom {
      box3f  domain;
      vec3f *vertices;
      float *scalars;
      umesh::UMesh::Tet *tets;
      int numTets;
#if UMESH_SHARED_FACES
      int2              *tetsOnFace;
#else
#endif
    };

    struct DD {
      inline __device__ bool sample(float &f, vec3f P, bool dbg) const;
      inline __device__ bool sampleElement(const int idx, float &f, vec3f P, bool dbg) const;
      /*! look up the given 3D (*local* world-space) point in the volume, and return the gradient */
      inline __device__ bool gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const;
      
      OptixTraversableHandle sampleAccel;
      box3f domain;
    };

    /*! build the given macro cell grid uses this volume's data */
    void buildMCs(MCGrid &mcGrid) override;
    
    void build(OWLContext owl,
               OWLModule owlDevCode) override;
    void setDD(OWLLaunchParams lp) override;
    void addLPVars(std::vector<OWLVarDecl> &lpVars) override;

    UMeshBrick::SP myBrick;
    UMesh::SP      umesh;
    
    DD        globals;
    
    OWLGeomType gt;
    OWLGeom     geom;
    OWLGroup    blas;
    OWLGroup    tlas;
#if UMESH_SHARED_FACES
    OWLBuffer   sharedFaceIndicesBuffer;
    OWLBuffer   sharedFaceNeighborsBuffer;
#else
#endif
    OWLBuffer   scalarsBuffer, tetsBuffer, verticesBuffer;
  };

  // ------------------------------------------------------------------
  
#ifdef __CUDA_ARCH__
  inline __device__
  bool UMeshVolume::DD::sample(float &f, vec3f P, bool dbg) const
  {
    SamplePRD prd;

    if (P.x >= domain.upper.x) return false;
    
    const float INVALID_VALUE = CUDART_INF;//1e20f;
    prd.sampledValue = INVALID_VALUE;
#if UMESH_SHARED_FACES
    owl::Ray sampleRay(P,vec3f(1.f,1e-6f,1e-6f),0.f,1e20f);
#else
    owl::Ray sampleRay(P,vec3f(1.f,1e-6f,1e-6f),0.f,1e-20f);
#endif
    traceRay(sampleAccel,sampleRay,prd);
    if (prd.sampledValue == INVALID_VALUE) return false;
    f = prd.sampledValue;
    return true;
  }

  inline __device__
  bool UMeshVolume::DD::gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const
  {
    float right,left,top,bottom,front,back;
    bool valid = true;
    valid &= sample(right, P+vec3f(delta.x,0.f,0.f),dbg);
    valid &= sample(left,  P-vec3f(delta.x,0.f,0.f),dbg);
    valid &= sample(top,   P+vec3f(0.f,delta.y,0.f),dbg);
    valid &= sample(bottom,P-vec3f(0.f,delta.y,0.f),dbg);
    valid &= sample(front, P+vec3f(0.f,0.f,delta.z),dbg);
    valid &= sample(back,  P-vec3f(0.f,0.f,delta.z),dbg);
    g = vec3f(right-left,top-bottom,front-back);
    return valid;
  }
#endif
  
}


