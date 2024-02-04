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

#include "owl/owl.h"
#if BARNEY_MPI
# include <mpi.h>
#endif

#define BN_API extern "C"

typedef struct _BNContext       *BNContext;
typedef struct _BNScalarField   *BNScalarField;
typedef struct _BNGeom          *BNGeom;
typedef struct _BNVolume        *BNVolume;
typedef struct _BNGroup         *BNGroup;
typedef struct _BNModel         *BNModel;
typedef struct _BNFrameBuffer   *BNFrameBuffer;
typedef struct _BNDataGroup     *BNDataGroup;
typedef struct _BNTexture2D     *BNTexture2D;
typedef BNTexture2D BNTexture;

typedef enum { BN_FLOAT, BN_UINT8 } BNScalarType;

struct BNMaterial {
  float3 baseColor;
  float  transparency;
  float  ior;
  BNTexture2D alphaTexture = 0;
  BNTexture2D colorTexture = 0;
};

/*! supported formats for texels in textures */
typedef enum {
  BN_TEXEL_FORMAT_RGBA8,
  BN_TEXEL_FORMAT_RGBA32F,
  BN_TEXEL_FORMAT_R8,
  BN_TEXEL_FORMAT_R32F
}
BNTexelFormat;

/*! currently supported texture filter modes */
typedef enum {
  BN_TEXTURE_NEAREST,
  BN_TEXTURE_LINEAR
}
BNTextureFilterMode;

/*! currently supported texture filter modes */
typedef enum {
  BN_TEXTURE_WRAP,
  BN_TEXTURE_CLAMP,
  BN_TEXTURE_BORDER,
  BN_TEXTURE_MIRROR
}
BNTextureAddressMode;

/*! Indicates if a texture is linear or SRGB */
typedef enum {
  BN_COLOR_SPACE_LINEAR,
  BN_COLOR_SPACE_SRGB
}
BNTextureColorSpace;


struct BNCamera {
  /*! vector from camera center to to lower-left pixel (i.e., pixel
    (0,0)) on the focal plane */
  float3 dir_00;
  /* vector along edge of image plane, in u direction */
  float3 dir_du;
  /* vector along edge of image plane, in v direction */
  float3 dir_dv;
  /*! lens center ... */
  float3 lens_00;
  /* vector along v direction, for ONE pixel */
  float  lensRadius;
};

#define BN_DEFAULT_MATERIAL  { \
    { .7f,.7f,.7f }, 0.f, 1.f, \
      /*no color tex*/0,       \
      /*no alpha tex*/0        \
      }

struct BNGridlet {
  float3 lower;
  float3 upper;
  int3   dims;
};

#define BN_FOVY_DEGREES(degrees) ((float)(degrees*M_PI/180.f))

BN_API
void bnPinholeCamera(BNCamera *camera,
                     float3 from,
                     float3 at,
                     float3 up,
                     float  fovy,
                     float  aspect);

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
void  bnBuild(BNDataGroup dataGroup);

// ==================================================================
// render interface
// ==================================================================

BN_API
void bnAccumReset(BNFrameBuffer fb);

BN_API
BNFrameBuffer bnFrameBufferCreate(BNContext context,
                                  int owningRank);
BN_API
void bnFrameBufferResize(BNFrameBuffer fb,
                         int sizeX, int sizeY,
                         uint32_t *hostRGBA,
                         float    *hostDepth = 0);

BN_API
BNDataGroup bnGetDataGroup(BNModel model,
                           int dataGroupID);

BN_API
void bnRender(BNModel model,
              const BNCamera *camera,
              BNFrameBuffer fb);

struct BNTransform {
  struct {
    struct {
      float3 vx;
      float3 vy;
      float3 vz;
    } l;
    float3 p;
  } xfm;
};

BN_API
void bnSetInstances(BNDataGroup dataGroup,
                    BNGroup *groupsToInstantiate,
                    BNTransform *instanceTransforms,
                    int numInstances);

BN_API
BNModel bnModelCreate(BNContext ctx);

// ==================================================================
// scene content
// ==================================================================

BN_API
BNGroup bnGroupCreate(BNDataGroup dataGroup,
                      BNGeom *geoms, int numGeoms,
                      BNVolume *volumes, int numVolumes);
BN_API
void bnGroupBuild(BNGroup group);

BN_API
BNTexture2D bnTexture2DCreate(BNDataGroup dataGroup,
                              BNTexelFormat texelFormat,
                              /*! number of texels in x dimension */
                              uint32_t size_x,
                              /*! number of texels in y dimension */
                              uint32_t size_y,
                              const void *texels,
                              BNTextureFilterMode  filterMode  = BN_TEXTURE_LINEAR,
                              BNTextureAddressMode addressMode = BN_TEXTURE_CLAMP,
                              BNTextureColorSpace  colorSpace  = BN_COLOR_SPACE_LINEAR);

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
void bnTriangleMeshUpdate(BNGeom geom,
                          const BNMaterial *material,
                          const int3 *indices,
                          int numIndices,
                          const float3 *vertices,
                          int numVertices,
                          const float3 *normals,
                          const float2 *texcoords);

BN_API
BNGeom bnSpheresCreate(BNDataGroup       dataGroup,
                       const BNMaterial *material,
                       const float3     *origins,
                       int               numSpheres,
                       /*! a per-sphere color that - if specified -
                           overwrites the material.baseColor; can be
                           null */
                       const float3     *colors,
                       const float      *radii,
                       float             defaultRadius);

/*! iw todo: split this into two different geometries: one for
    'cylinders', and one for 'rounded cones' */
BN_API
BNGeom bnCylindersCreate(BNDataGroup       dataGroup,
                         const BNMaterial *material,
                         const float3     *points,
                         int               numPoints,
                         const float3     *colors,
                         /*! if true - and colors is non null - then
                             the colors array specifies per-vertex
                             colors */
                         bool              colorPerVertex,
                         const int2       *indices,
                         int               numIndices,
                         const float      *radii,
                         /*! if true - and radii is non null -then the
                             radii specify per-vertex radii and
                             segments will be rounded cones */
                         bool              radiusPerVertex,
                         float             defaultRadius);

BN_API
void bnGeomSetMaterial(BNGeom geom, BNMaterial *material);

// ------------------------------------------------------------------
// volume stuff
// ------------------------------------------------------------------

BN_API
BNScalarField bnStructuredDataCreate(BNDataGroup dataGroup,
                                     int3 dims,
                                     BNScalarType type,
                                     const void *scalars,
                                     float3 gridOrigin,
                                     float3 gridSpacing);

BN_API
BNScalarField bnUMeshCreate(BNDataGroup dataGroup,
                            // vertices, 4 floats each (3 floats position,
                            // 4th float scalar value)
                            const float *vertices, int numVertices,
                            // tets, 4 ints in vtk-style each
                            const int *tets,       int numTets,
                            // pyramids, 5 ints in vtk-style each
                            const int *pyrs,       int numPyrs,
                            // wedges/tents, 6 ints in vtk-style each
                            const int *wedges,     int numWedges,
                            // general (non-guaranteed cube/voxel) hexes, 8
                            // ints in vtk-style each
                            const int *hexes,      int numHexes,
                            //
                            int numGrids,
                            // offsets into gridIndices array
                            const int *_gridOffsets,
                            // grid dims (3 floats each)
                            const int *_gridDims,
                            // grid domains, 6 floats each (3 floats min corner,
                            // 3 floats max corner)
                            const float *gridDomains,
                            // grid scalars
                            const float *gridScalars,
                            int numGridScalars,
                            const float3 *domainOrNull=0);


BN_API
BNScalarField bnBlockStructuredAMRCreate(BNDataGroup dataGroup,
                                         /*TODO:const float *cellWidths,*/
                                         // block bounds, 6 ints each (3 for min,
                                         // 3 for max corner)
                                         const int *blockBounds, int numBlocks,
                                         // refinement level, per block,
                                         // finest is level 0,
                                         const int *blockLevels,
                                         // offsets into blockData array
                                         const int *blockOffsets,
                                         // block scalars
                                         const float *blockScalars, int numBlockScalars);


BN_API
BNVolume bnVolumeCreate(BNDataGroup dataGroup,
                        BNScalarField sf);

BN_API
void bnVolumeSetXF(BNVolume volume,
                   float2 domain,
                   const float4 *colorMap,
                   int numColorMapValues,
                   float densityAt1);

