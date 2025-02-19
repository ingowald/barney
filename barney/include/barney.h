// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#define BARNEY_VERSION_MAJOR @BARNEY_VERSION_MAJOR@
#define BARNEY_VERSION_MINOR @BARNEY_VERSION_MINOR@
#define BARNEY_VERSION_PATCH @BARNEY_VERSION_PATCH@

#cmakedefine01 BARNEY_BACKEND_OPTIX
#cmakedefine01 BARNEY_BACKEND_CUDA
#cmakedefine01 BARNEY_BACKEND_EMBREE

#include <cstdint>
#include <cstddef>
#if BARNEY_MPI
# include <mpi.h>
#endif

#if defined(_MSC_VER)
#  define BARNEY_DLL_EXPORT __declspec(dllexport)
#  define BARNEY_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define BARNEY_DLL_EXPORT __attribute__((visibility("default")))
#  define BARNEY_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define BARNEY_DLL_EXPORT
#  define BARNEY_DLL_IMPORT
#endif

#ifdef __cplusplus
#  define BARNEY_API extern "C" BARNEY_DLL_EXPORT
#else
#  define BARNEY_API /* bla */
#endif

typedef struct _BNContext                           *BNContext;
typedef struct _BNObject                         {} *BNObject;
typedef struct _BNData         : public _BNObject{} *BNData;
typedef struct _BNTextureData  : public _BNObject{} *BNTextureData;
typedef struct _BNScalarField  : public _BNObject{} *BNScalarField;
typedef struct _BNGeom         : public _BNObject{} *BNGeom;
typedef struct _BNVolume       : public _BNObject{} *BNVolume;
typedef struct _BNGroup        : public _BNObject{} *BNGroup;
typedef struct _BNModel        : public _BNObject{} *BNModel;
typedef struct _BNRenderer     : public _BNObject{} *BNRenderer;
typedef struct _BNFrameBuffer  : public _BNObject{} *BNFrameBuffer;
// typedef struct _BNDataGroup    : public _BNObject{} *BNDataGroup;
typedef struct _BNTexture2D    : public _BNObject{} *BNTexture2D;
typedef struct _BNTexture3D    : public _BNObject{} *BNTexture3D;
typedef struct _BNLight        : public _BNObject{} *BNLight;
typedef struct _BNCamera       : public _BNObject{} *BNCamera;
typedef struct _BNMaterial     : public _BNObject{} *BNMaterial;
typedef struct _BNSampler     : public _BNObject{} *BNSampler;

typedef BNTexture2D BNTexture;

struct bn_float2 { float x,y; };
struct bn_float3 { float x,y,z; };
struct bn_float4 { float x,y,z,w; };
struct bn_int2 { int x,y; };
struct bn_int3 { int x,y,z; };
struct bn_int4 { int x,y,z,w; };

typedef enum {
  /*! a undefined data type */
  BN_DATA_UNDEFINED,
  /*! BNData */
  BN_DATA,
  /*! BNObject */
  BN_OBJECT,
  /*! BNTexture */
  BN_TEXTURE,
  BN_TEXTURE_3D,
  /*! int32_t */
  BN_INT,
  /*! int2 */
  BN_INT2,
  BN_INT3,
  BN_INT4,
  BN_FLOAT,
  BN_FLOAT2,
  BN_FLOAT3,
  BN_FLOAT4,

  BN_UFIXED8,
  BN_UFIXED8_RGBA,
  BN_UFIXED8_RGBA_SRGB,

  BN_UFIXED16,
  
  BN_FLOAT4_RGBA,
  
  BN_RAW_DATA_BASE
} BNDataType;

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

struct BNGridlet {
  bn_float3 lower;
  bn_float3 upper;
  bn_int3   dims;
};

#define BN_FOVY_DEGREES(degrees) ((float)(degrees*M_PI/180.f))




// ==================================================================
// creators for CONTEXT-owned objects
// ==================================================================

/*! create a new camera of given type. currently supported types:
    "pinhole" */
BARNEY_API
BNCamera bnCameraCreate(BNContext context,
                        const char *type);

BARNEY_API
BNFrameBuffer bnFrameBufferCreate(BNContext context,
                                  int owningRank);

BARNEY_API
BNModel bnModelCreate(BNContext ctx);
BARNEY_API
BNRenderer bnRendererCreate(BNContext ctx, const char *ignoreForNow);




// ==================================================================
// general set/commit semantics
// ==================================================================

BARNEY_API
void bnCommit(BNObject target);
              
BARNEY_API
void bnSetString(BNObject target, const char *paramName, const char *value);

BARNEY_API
void bnSetData(BNObject target, const char *paramName, BNData value);

BARNEY_API
void bnSetObject(BNObject target, const char *paramName, const BNObject value);

BARNEY_API
void bnSetLight(BNObject target, const char *paramName, BNLight value);

BARNEY_API
void bnSet1i(BNObject target, const char *paramName, int value);

BARNEY_API
void bnSet2i(BNObject target, const char *paramName, int x, int y);

BARNEY_API
void bnSet3i(BNObject target, const char *paramName, int x, int y, int z);

BARNEY_API
void bnSet4i(BNObject target, const char *paramName, int x, int y, int z, int w);

BARNEY_API
void bnSet1f(BNObject target, const char *paramName, float value);

BARNEY_API
void bnSet2f(BNObject target, const char *paramName, float x, float y);

BARNEY_API
void bnSet3f(BNObject target, const char *paramName, float x, float y, float z);

BARNEY_API
void bnSet4f(BNObject target, const char *paramName, float x, float y, float z, float w);

BARNEY_API
void bnSet4x3fv(BNObject target, const char *paramName, const float *affineMatrix);

BARNEY_API
void bnSet4x4fv(BNObject target, const char *paramName, const float *xfm);

/* add cuda vector type variants of set functions; but do that only if
   cuda.h or cuda/vector_types.h has already been included */
# ifdef __VECTOR_TYPES__
BARNEY_API
void bnSet2ic(BNObject target, const char *paramName, int2 v);

BARNEY_API
void bnSet3ic(BNObject target, const char *paramName, int3 v);

BARNEY_API
void bnSet4ic(BNObject target, const char *paramName, int4 v);

BARNEY_API
void bnSet2fc(BNObject target, const char *paramName, float2 v);

BARNEY_API
void bnSet3fc(BNObject target, const char *paramName, float3 v);

BARNEY_API
void bnSet4fc(BNObject target, const char *paramName, float4 v);
# endif


/*! helper function to fill in a BNCamera structure from a more
    user-friendly from/at/up/fovy specification */
BARNEY_API
void bnPinholeCamera(BNCamera  camera,
                     bn_float3 from,
                     bn_float3 at,
                     bn_float3 up,
                     float     fovy,
                     float     aspect);

BARNEY_API
BNContext bnContextCreate(/*! how many data slots this context is to
                              offer, and which part(s) of the
                              distributed model data these slot(s)
                              will hold */
                          const int *dataRanksOnThisContext=0,
                          int        numDataRanksOnThisContext=1,
                          /*! which gpu(s) to use for this
                            process. default is to distribute
                            node's GPUs equally over all ranks on
                             that given node */
                          const int *gpuIDs=nullptr,
                          int  numGPUs=-1);

BARNEY_API
void bnCountAvailableDevice(int *numGPUs);

BARNEY_API
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
BARNEY_API
BNContext bnMPIContextCreate(MPI_Comm comm,
                             /*! how many data slots this context is to
                               offer, and which part(s) of the
                               distributed model data these slot(s)
                               will hold */
                             const int *dataRanksOnThisContext=0,
                             int        numDataRanksOnThisContext=1,
                             /*! which gpu(s) to use for this
                               process. default is to distribute
                               node's GPUs equally over all ranks on
                               that given node */
                             const int *gpuIDs=nullptr,
                             int  numGPUs=-1
                             );

BARNEY_API
void  bnMPIQueryHardware(BNHardwareInfo *hardware, MPI_Comm comm);

#endif


/*! decreases (the app's) reference count of said object by one. if
    said refernce count falls to 0 the object handle gets destroyed
    and may no longer be used by the app, and the object referenced to
    by this handle may be removed (from the app's point of view). Note
    the object referenced by this handle may not get destroyed
    immediagtely if it had other indirect references, such as, for
    example, a group still holding a refernce to a geometry */
BARNEY_API
void  bnRelease(BNObject object);

/*! increases (the app's) reference count of said object byb one */
BARNEY_API
void  bnAddReference(BNObject object);

BARNEY_API
void  bnBuild(BNModel model, int whichDataSlot);

// ==================================================================
// render interface
// ==================================================================

BARNEY_API
void bnAccumReset(BNFrameBuffer fb);

typedef enum {
  BN_FB_COLOR = (1<<0),
  BN_FB_DEPTH = (1<<1),
} BNFrameBufferChannel;

BARNEY_API
void bnFrameBufferResize(BNFrameBuffer fb,
                         int sizeX, int sizeY,
                         uint32_t requiredChannels = BN_FB_COLOR);

BARNEY_API
void bnFrameBufferRead(BNFrameBuffer fb,
                       BNFrameBufferChannel channelToRead,
                       void *pointerToReadDataInto,
                       BNDataType requiredFormat);

BARNEY_API
void *bnFrameBufferGetPointer(BNFrameBuffer fb,
                              BNFrameBufferChannel channelToRead);

BARNEY_API
void bnRender(BNRenderer    renderer,
              BNModel       model,
              BNCamera      camera,
              BNFrameBuffer fb);

struct BNTransform {
  struct {
    bn_float3 vx;
    bn_float3 vy;
    bn_float3 vz;
  } l;
  bn_float3 p;
};

BARNEY_API
void bnSetInstances(BNModel model,
                    int whichSlot,
                    BNGroup *groupsToInstantiate,
                    BNTransform *instanceTransforms,
                    int numInstances);

// ==================================================================
// scene content
// ==================================================================

/*! a regular, one-dimensional array of numItems elements of given
  type. On the device the respective elemtns will appear as a plain
  CUDA array that lies in either device or managed memory.  Note data
  arrays of this type can _not_ be assigned to samplers because these
  need data to be put into cudaArray's (in order to create
  cudaTextures) */
BARNEY_API
BNData bnDataCreate(BNContext context,
                    int whichSlot,
                    BNDataType dataType,
                    size_t numItems,
                    const void *items);

// /*! creates a cudaArray2D of specified size and texels. Can be passed
//   to a sampler to create a matching cudaTexture2D */
// BARNEY_API
// BNTextureData bnTextureData2DCreate(BNContext context,
//                                     int whichSlot,
//                                     BNTexelFormat texelFormat,
//                                     int width, int height,
//                                     const void *items);

/*! creates a cudaArray2D of specified size and texels. Can be passed
  to a sampler to create a matching cudaTexture2D, or as a background
  image to a renderer */
BARNEY_API
BNTextureData bnTextureData2DCreate(BNContext context,
                                    int whichSlot,
                                    BNDataType texelFormat,
                                    int width, int height,
                                    const void *items);

BARNEY_API
BNLight bnLightCreate(BNContext context,
                      int whichSlot,
                      const char *type);
                    
BARNEY_API
BNGroup bnGroupCreate(BNContext context,
                      int whichSlot,
                      BNGeom *geoms, int numGeoms,
                      BNVolume *volumes, int numVolumes);
BARNEY_API
void bnGroupBuild(BNGroup group);

BARNEY_API
BNTexture2D bnTexture2DCreate(BNContext context,
                              int whichSlot,
                              BNDataType texelFormat,
                              /*! number of texels in x dimension */
                              uint32_t size_x,
                              /*! number of texels in y dimension */
                              uint32_t size_y,
                              const void *texels,
                              BNTextureFilterMode  filterMode  = BN_TEXTURE_LINEAR,
                              BNTextureAddressMode addressMode_x = BN_TEXTURE_WRAP,
                              BNTextureAddressMode addressMode_y = BN_TEXTURE_WRAP,
                              BNTextureColorSpace  colorSpace  = BN_COLOR_SPACE_LINEAR);

BARNEY_API
BNTexture3D bnTexture3DCreate(BNContext context,
                              int whichSlot,
                              BNDataType texelFormat,
                              /*! number of texels in x dimension */
                              uint32_t size_x,
                              /*! number of texels in y dimension */
                              uint32_t size_y, 
                              /*! number of texels in z dimension */
                              uint32_t size_z,
                              const void *texels,
                              BNTextureFilterMode  filterMode  = BN_TEXTURE_LINEAR,
                              BNTextureAddressMode addressMode = BN_TEXTURE_CLAMP);

// ------------------------------------------------------------------
// object-"create" interface
// ------------------------------------------------------------------

/*! create a new geometry of given type. currently supported types:
    "triangles", "spheres", "cylinders" */
BARNEY_API
BNGeom bnGeometryCreate(BNContext context,
                        int whichSlot,
                        const char *type);


/*! create a new scalar field of given type. currently supported
    types: "structured" */
BARNEY_API
BNScalarField bnScalarFieldCreate(BNContext context,
                                  int whichSlot,
                                  const char *type);
                                     
/*! create a new material of given type. currently supported types:
  "matte", "glass" */
BARNEY_API
BNMaterial bnMaterialCreate(BNContext context,
                            int whichSlot,
                            const char *type);

BARNEY_API
BNSampler bnSamplerCreate(BNContext context,
                          int whichSlot,
                          const char *type);









// // ------------------------------------------------------------------
// // soon to be deprecated, but still the only way to create those
// // ------------------------------------------------------------------
// BARNEY_API
// BNScalarField bnUMeshCreate(BNContext context,
//                             int whichSlot,
//                             // vertices, 4 floats each (3 floats position,
//                             // 4th float scalar value)
//                             const float4 *vertices, int numVertices,
//                             /*! array of all the vertex indices of all
//                                 elements, one after another;
//                                 ie. elements with different vertex
//                                 counts can come in any order, so a
//                                 mesh with one tet and one hex would
//                                 have an index array of size 12, with
//                                 four for the tet and eight for the
//                                 hex */
//                             const int *_indices, int numIndices,
//                             /*! one int per logical element, stating
//                                 where in the indices array it's N
//                                 differnt vertices will be located */
//                             const int *_elementOffsets,
//                             int numElements,
//                             // // tets, 4 ints in vtk-style each
//                             // const int *tets,       int numTets,
//                             // // pyramids, 5 ints in vtk-style each
//                             // const int *pyrs,       int numPyrs,
//                             // // wedges/tents, 6 ints in vtk-style each
//                             // const int *wedges,     int numWedges,
//                             // // general (non-guaranteed cube/voxel) hexes, 8
//                             // // ints in vtk-style each
//                             // const int *hexes,      int numHexes,
//                             // //
//                             // int numGrids,
//                             // // offsets into gridIndices array
//                             // const int *_gridOffsets,
//                             // // grid dims (3 floats each)
//                             // const int *_gridDims,
//                             // // grid domains, 6 floats each (3 floats min corner,
//                             // // 3 floats max corner)
//                             // const float *gridDomains,
//                             // // grid scalars
//                             // const float *gridScalars,
//                             // int numGridScalars,
//                             const bn_float3 *domainOrNull=0);


// BARNEY_API
// BNScalarField bnBlockStructuredAMRCreate(BNContext context,
//                                          int whichSlot,
//                                          /*TODO:const float *cellWidths,*/
//                                          // block bounds, 6 ints each (3 for min,
//                                          // 3 for max corner)
//                                          const int *blockBounds, int numBlocks,
//                                          // refinement level, per block,
//                                          // finest is level 0,
//                                          const int *blockLevels,
//                                          // offsets into blockData array
//                                          const int *blockOffsets,
//                                          // block scalars
//                                          const float *blockScalars, int numBlockScalars);


BARNEY_API
BNVolume bnVolumeCreate(BNContext context,
                        int whichSlot,
                        BNScalarField sf);

BARNEY_API
void bnVolumeSetXF(BNVolume         volume,
                   bn_float2        domain,
                   const bn_float4 *colorMap,
                   int              numColorMapValues,
                   float            densityAt1);





// ==================================================================
// HELPER FUNCTION(S) - may not survivie into final API
// ==================================================================

#ifdef __cplusplus
inline void bnSetAndRelease(BNObject o, const char *n, BNObject v)
{
  bnSetObject(o,n,v);
  bnRelease(v);
}
inline void bnSetAndRelease(BNObject o, const char *n, BNData v)
{
  bnSetData(o,n,v);
  bnRelease(v);
}
inline void bnSet(BNObject o, const char *n, bn_float2 v)
{ bnSet2f(o,n,v.x,v.y); }
inline void bnSet(BNObject o, const char *n, bn_float3 v)
{ bnSet3f(o,n,v.x,v.y,v.z); }
inline void bnSet(BNObject o, const char *n, bn_float4 v)
{ bnSet4f(o,n,v.x,v.y,v.z,v.w); }
inline void bnSet(BNObject o, const char *n, bn_int2 v)
{ bnSet2i(o,n,v.x,v.y); }
inline void bnSet(BNObject o, const char *n, bn_int3 v)
{ bnSet3i(o,n,v.x,v.y,v.z); }
inline void bnSet(BNObject o, const char *n, bn_int4 v)
{ bnSet4i(o,n,v.x,v.y,v.z,v.w); }
#endif

// struct BNMaterialHelper {
//   bn_float3   baseColor     { .7f,.7f,.7f };
//   float       transmission  { 0.f };
//   float       ior           { 1.45f };
//   float       metallic      { 1.f };
//   float       roughness     { 0.f };
//   BNTexture2D alphaTexture  { 0 };
//   BNTexture2D colorTexture  { 0 };
// };

// /*! c++ helper function */
// inline void bnSetAndRelease(BNObject target, const char *paramName,
//                             BNObject value)
// { bnSetObject(target,paramName,value); bnRelease(value); }

// /*! c++ helper function */
// inline void bnSetAndRelease(BNObject target, const char *paramName,
//                             BNData value)
// { bnSetData(target,paramName,value); bnRelease(value); }
  
// /*! helper function for assinging leftover BNMaterial definition from old API */
// inline void bnAssignMaterial(BNGeom geom,const BNMaterialHelper *material)
// {
//   bnSet3fc(geom,"material.baseColor",material->baseColor);
//   bnSet1f(geom,"material.transmission",material->transmission);
//   bnSet1f(geom,"material.ior",material->ior);
//   if (material->colorTexture)
//     bnSetObject(geom,"material.colorTexture",material->colorTexture);
//   if (material->alphaTexture)
//     bnSetObject(geom,"material.alphaTexture",material->alphaTexture);
//   bnCommit(geom);
// }




// // ------------------------------------------------------------------
// // DEPRECATED
// // ------------------------------------------------------------------
// BARNEY_API
// BNGeom bnTriangleMeshCreate(BNContext context,
//                             int whichSlot,
//                             const BNMaterialHelper *material,
//                             const int3 *indices,
//                             int numIndices,
//                             const float3 *vertices,
//                             int numVertices,
//                             const float3 *normals,
//                             const float2 *texcoords);

// // ------------------------------------------------------------------
// // DEPRECATED
// // ------------------------------------------------------------------
// BARNEY_API
// BNScalarField bnStructuredDataCreate(BNContext context,
//                                      int whichSlot,
//                                      int3 dims,
//                                      BNDataType /*ScalarType*/ type,
//                                      const void *scalars,
//                                      float3 gridOrigin,
//                                      float3 gridSpacing);

