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

#pragma once

#include "owl/owl_host.h"
#if BARNEY_MPI
# include <mpi.h>
#endif

#define BN_API extern "C"
typedef struct _BNContext *BNContext;
typedef struct _BNGeom    *BNGeom;
typedef struct _BNVolume  *BNVolume;
typedef struct _BNGroup   *BNGroup;
typedef struct _BNXF      *BNXF;
typedef struct _BNModel   *BNModel;

struct BNMaterial {
  float3 diffuseColor;
  float  transparency;
  float  ior;
  int    alphaTextureID;
  int    colorTextureID;
};

#define BN_DEFAULT_MATERIAL  { { .5f,.5f,.5f }, 0.f, 1.f, -1, -1 }

struct BNGridlet {
  float3 lower;
  float3 upper;
  int3   dims;
};

BN_API
BNContext *bnContextCreate(int *gpuIDs=nullptr,
                           int numGPUs=-1
                           /* int numDifferentDataGroups=1 ? */
                           /*, bool replicated_vs_distributed_mode? */
                           );

BN_API
void bnContextDestroy(BNContext context);

struct BNHardwareInfo {
  int gpusOnNode;
  int ranksOnNode;
  int gpusOnRank;
};


#if BARNEY_MPI
BN_API
BNContext bnContextCreateMPI(MPI_Comm comm,
                             /*! which data group(s) this rank will
                                 owl - default is 1 group, with data
                                 group equal to mpi rank */
                             int *dataGroupsOnThisRank=0,
                             int  numDataGroupsOnThisRank=1,
                             /*! which gpu(s) to use for this
                                 process. default is to distribute
                                 node's GPUs equally over all ranks on
                                 that given node */
                             int *gpuIDs=nullptr,
                             int  numGPUs=-1
                             );

BN_API
void  bnQueryHardwareMPI(BNHardwareInfo *hardware, MPI_Comm comm);


#endif


// ==================================================================
// render interface
// ==================================================================

typedef struct _BNFB *BNFB;
void bnFBCreate(BNContext context, int format);
void bnFBResize(BNFB fb, int sizeX, int sizeY);

struct BNRenderRequest {
  struct {
    float3 center;
    float3 uAxis;
    float3 vAxis;
  } focalPlane;
  struct {
    float3 at;
    float3 du;
    float3 dv;
  } lens;
  struct {
    struct {
      float2 lower = { 0.f, 0.f };
      float2 upper = { 1.f, 1.f };
    } region;
    int  spp   = 1;
    int  seed  = 0;
  } frame;
};

BN_API
BNModel bnModelCreate(BNContext ctx,
                      int dataGroupIdx,
                      BNGroup *groups,
                      int numGroups,
                      BNVolume *volumes,
                      int numVolumes);

BN_API
void bnRender(BNModel model,
              BNRenderRequest *rr,
              BNFB fb);

// ==================================================================
// scene content
// ==================================================================

BN_API
BNGroup bnGroupCreate(BNContext context,
                      int dataGroupIdx,
                      BNGeom *geoms, int numGeoms,
                      BNVolume *volumes, int numVolumes);


// ------------------------------------------------------------------
// geometry stuff
// ------------------------------------------------------------------

BN_API
BNGeom bnTriangleMeshCreate(BNContext context,
                            int dataGroupIdx,
                            BNMaterial *material,
                            int3 *indices,
                            int numIndices,
                            float3 *vertices,
                            int numVertices,
                            float3 *normals,
                            float2 *texcoords);

BN_API
void bnGeomSetMaterial(BNGeom geom, BNMaterial *material);
// ------------------------------------------------------------------
// volume stuff
// ------------------------------------------------------------------

BN_API
BNXF bnXFCreate(BNContext context,
                int numScalars);
BN_API
void bnXFSet(float domain_lower, float domain_upper,
             float4 *values,
             float densityAt1);


BN_API
BNVolume bnUMeshCreate(BNContext context,
                       float3 *vertices,
                       int numVertices,
                       BNXF xf);

BN_API
void bnUMeshSetScalars(BNVolume umesh,
                       float *scalars);

BN_API
void bnUMeshSetTets(BNVolume umesh,
                    int *indices,
                    int numTets);

BN_API
void bnUMeshSetHexes(BNVolume umesh,
                     int *indices,
                     int numHexes);

BN_API
void bnUMeshSetGridlets(BNVolume umesh,
                        int *indices,
                        int numIndices,
                        BNGridlet *gridlets,
                        int numGridlets);

                         

