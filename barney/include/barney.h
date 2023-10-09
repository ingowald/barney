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

typedef struct _BNContext       *BNContext;
typedef struct _BNGeom          *BNGeom;
typedef struct _BNVolume        *BNVolume;
typedef struct _BNGroup         *BNGroup;
typedef struct _BNXF            *BNXF;
typedef struct _BNModel         *BNModel;
typedef struct _BNRenderRequest *BNRenderRequest;
typedef struct _BNFrameBuffer   *BNFrameBuffer;
typedef struct _BNDataGroup     *BNDataGroup;

struct BNMaterial {
  float3 diffuseColor;
  float  transparency;
  float  ior;
  int    alphaTextureID;
  int    colorTextureID;
};

struct BNCamera {
  float3 lensCenter;
  float  lensRadius;
  float3 focalPoint;
  float  fovy;
  float  aspect;
  struct {
    float2 lower;
    float2 upper;
  } imageRegion;
};

#define BN_DEFAULT_MATERIAL  { { .5f,.5f,.5f }, 0.f, 1.f, -1, -1 }

struct BNGridlet {
  float3 lower;
  float3 upper;
  int3   dims;
};

#define BN_FOVY_DEGREES(degrees) ((float)(degrees*M_PI/180.f))

BN_API
void bnPinholeCamera(BNCamera *camera,
                     float from_x,
                     float from_y,
                     float from_z,
                     float at_x,
                     float at_y,
                     float at_z,
                     float up_x,
                     float up_y,
                     float up_z,
                     float fovy,
                     float aspect);

BN_API
BNContext bnContextCreate(/*! which data group(s) this rank will
                            owl - default is 1 group, with data
                            group equal to mpi rank */
                          const int *dataGroupsOnThisRank=0,
                          int  numDataGroupsOnThisRank=1,
                          /*! which gpu(s) to use for this
                            process. default is to distribute
                            node's GPUs equally over all ranks on
                             that given node */
                          const int *gpuIDs=nullptr,
                          int  numGPUs=-1);

BN_API
void bnContextDestroy(BNContext context);

struct BNHardwareInfo {
  int numRanks;
  int numHosts;
  int numGPUsThisRank;
  int numGPUsThisHost;
  int numRanksThisHost;
  int localRank;
};


#if BARNEY_MPI
BN_API
BNContext bnMPIContextCreate(MPI_Comm comm,
                             /*! which data group(s) this rank will
                               own - default is 1 group, with data
                                 group equal to mpi rank */
                             const int *dataGroupsOnThisRank=0,
                             int  numDataGroupsOnThisRank=1,
                             /*! which gpu(s) to use for this
                                 process. default is to distribute
                                 node's GPUs equally over all ranks on
                                 that given node */
                             const int *gpuIDs=nullptr,
                             int  numGPUs=-1
                             );

BN_API
void  bnMPIQueryHardware(BNHardwareInfo *hardware, MPI_Comm comm);

#endif

BN_API
void  bnModelBuild(BNModel model);

// ==================================================================
// render interface
// ==================================================================

BN_API
BNFrameBuffer bnFrameBufferCreate(BNContext context,
                                  int owningRank);
BN_API
void bnFrameBufferResize(BNFrameBuffer fb,
                         int sizeX, int sizeY,
                         uint32_t *hostRGBA);

// void bnFBResize(BNFrameBuffer fb, int sizeX, int sizeY);

// struct BNRenderJob {
//   struct {
//     float3 center;
//     float3 uAxis;
//     float3 vAxis;
//   } focalPlane;
//   struct {
//     float3 at;
//     float3 du;
//     float3 dv;
//   } lens;
//   struct {
//     struct {
//       float2 lower = { 0.f, 0.f };
//       float2 upper = { 1.f, 1.f };
//     } region;
//     int  spp   = 1;
//     int  seed  = 0;
//   } frame;
// };

BN_API
BNDataGroup bnGetDataGroup(BNModel model,
                           int dataGroupID);

BN_API
void bnRender(BNModel model,
              const BNCamera *camera,
              BNFrameBuffer fb,
              BNRenderRequest *req);

BN_API
void bnModelSetInstances(BNDataGroup dataGroup,
                         BNGroup *groups,
                         int numGroups,
                         BNVolume *volumes,
                         int numVolumes);

BN_API
BNModel bnModelCreate(BNContext ctx);

// BN_API
// void bnRender(BNModel model,
//               BNRenderRequest *rr,
//               BNFrameBuffer fb);

// ==================================================================
// scene content
// ==================================================================

BN_API
BNGroup bnGroupCreate(BNDataGroup dataGroup,
                      BNGeom *geoms, int numGeoms,
                      BNVolume *volumes, int numVolumes);


// ------------------------------------------------------------------
// geometry stuff
// ------------------------------------------------------------------

BN_API
BNGeom bnTriangleMeshCreate(BNDataGroup dataGroup,
                            const BNMaterial *material,
                            const int3 *indices,
                            int numIndices,
                            const float3 *vertices,
                            int numVertices,
                            const float3 *normals,
                            const float2 *texcoords);

BN_API
void bnGeomSetMaterial(BNGeom geom, BNMaterial *material);
// ------------------------------------------------------------------
// volume stuff
// ------------------------------------------------------------------

BN_API
BNXF bnXFCreate(BNDataGroup dataGroup,
                int numScalars);
BN_API
void bnXFSet(float domain_lower, float domain_upper,
             const float4 *values,
             float densityAt1);


BN_API
BNVolume bnUMeshCreate(BNContext context,
                       const float3 *vertices,
                       int numVertices,
                       BNXF xf);

BN_API
void bnUMeshSetScalars(BNVolume umesh,
                       const float *scalars);

BN_API
void bnUMeshSetTets(BNVolume umesh,
                    const int *indices,
                    int numTets);

BN_API
void bnUMeshSetHexes(BNVolume umesh,
                     const int *indices,
                     int numHexes);

BN_API
void bnUMeshSetGridlets(BNVolume umesh,
                        const int *indices,
                        int numIndices,
                        const BNGridlet *gridlets,
                        int numGridlets);

                         

