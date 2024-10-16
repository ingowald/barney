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

#include "barney.h"
#include "barney/Context.h"
#include "barney/LocalContext.h"
#include "barney/fb/FrameBuffer.h"
#include "barney/GlobalModel.h"
#include "barney/geometry/Triangles.h"
#include "barney/volume/ScalarField.h"
#include "barney/umesh/common/UMeshField.h"
#include "barney/common/Data.h"
#include "barney/common/mat4.h"
#include "barney/Camera.h"

#define WARN_NOTIMPLEMENTED std::cout << " ## " << __PRETTY_FUNCTION__ << " not implemented yet ..." << std::endl;

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif

namespace barney {

  inline Context *checkGet(BNContext context)
  {
    assert(context);
    return (Context *)context;
  }
  
  inline std::string checkGet(const char *s)
  {
    assert(s != nullptr);
    return s;
  }
  
  inline Object *checkGet(BNObject object)
  {
    assert(object);
    return (Object *)object;
  }

  inline Data *checkGet(BNData data)
  {
    assert(data);
    return (Data *)data;
  }

  inline Camera *checkGet(BNCamera camera)
  {
    assert(camera);
    return (Camera *)camera;
  }
  
  inline GlobalModel *checkGet(BNModel model)
  {
    assert(model);
    return (GlobalModel *)model;
  }
  
  inline ModelSlot *checkGet(BNModel model, int whichSlot)
  {
    return checkGet(model)->getSlot(whichSlot);
  }
  
  inline Geometry *checkGet(BNGeom geom)
  {
    assert(geom);
    return (Geometry *)geom;
  }
  
  inline Volume *checkGet(BNVolume volume)
  {
    assert(volume);
    return (Volume *)volume;
  }
  inline ScalarField *checkGet(BNScalarField sf)
  {
    assert(sf);
    return (ScalarField *)sf;
  }
  
  inline Group *checkGet(BNGroup group)
  {
    assert(group);
    return (Group *)group;
  }
  
  
  inline FrameBuffer *checkGet(BNFrameBuffer frameBuffer)
  {
    assert(frameBuffer);
    return (FrameBuffer *)frameBuffer;
  }

  // ------------------------------------------------------------------

  inline Group::SP checkGetSP(BNGroup group)
  {
    return checkGet(group)->shared_from_this()->as<Group>();
  }
  inline ScalarField::SP checkGetSP(BNScalarField sf)
  {
    return checkGet(sf)->shared_from_this()->as<ScalarField>();
  }

  // ------------------------------------------------------------------

  BN_API
  BNTexture2D bnTexture2DCreate(BNModel model,
                                int whichSlot,
                                BNTexelFormat texelFormat,
                                /*! number of texels in x dimension */
                                uint32_t size_x,
                                /*! number of texels in y dimension */
                                uint32_t size_y,
                                const void *texels,
                                BNTextureFilterMode  filterMode,
                                BNTextureAddressMode addressMode,
                                BNTextureColorSpace  colorSpace)
  {
    LOG_API_ENTRY;
    Texture *texture = checkGet(model,whichSlot)->createTexture
      (texelFormat,vec2i(size_x,size_y),texels,
       filterMode,addressMode,colorSpace);
    return (BNTexture2D)texture;
  }

  BN_API
  BNTexture3D bnTexture3DCreate(BNModel model,
                                int whichSlot,
                                BNTexelFormat texelFormat,
                                uint32_t size_x,
                                uint32_t size_y,
                                uint32_t size_z,
                                const void *texels,
                                BNTextureFilterMode  filterMode,
                                BNTextureAddressMode addressMode)
  {
    LOG_API_ENTRY;
    Texture3D::SP tex
      = std::make_shared<Texture3D>
      (checkGet(model,whichSlot),
       texelFormat,vec3i(size_x,size_y,size_z),texels,
       filterMode,addressMode);
    checkGet(model,whichSlot)->context->initReference(tex);
    return (BNTexture3D)tex.get();
  }
  
  // ------------------------------------------------------------------
  
  BN_API
  BNModel bnModelCreate(BNContext ctx)
  {
    LOG_API_ENTRY;
    return (BNModel)checkGet(ctx)->createModel();
  }

  BN_API
  void bnSetInstances(BNModel model,
                      int whichSlot,
                      BNGroup *_groups,
                      BNTransform *xfms,
                      int numInstances)
  {
    LOG_API_ENTRY;

    std::vector<Group::SP> groups;
    for (int i=0;i<numInstances;i++) {
      groups.push_back(checkGetSP(_groups[i]));
    }
    checkGet(model,whichSlot)->setInstances(groups,(const affine3f *)xfms);
  }
  
  BN_API
  void  bnRelease(BNObject _object)
  {
    LOG_API_ENTRY;
    Object *object = checkGet(_object);
    assert(object);
    Context *context = object->getContext();
    assert(context);
    context->releaseHostReference(object->shared_from_this());
  }
  
  BN_API
  void  bnAddReference(BNObject _object)
  {
    LOG_API_ENTRY;
    if (_object == 0) return;
    Object *object = checkGet(_object);
    Context *context = object->getContext();
    context->addHostReference(object->shared_from_this());
  }


  BN_API
  void bnContextDestroy(BNContext context)
  {
    LOG_API_ENTRY;
    delete (Context *)context;
  }

  BN_API
  BNGeom bnTriangleMeshCreate(BNModel model,
                              int whichSlot,
                              const BNMaterialHelper *material,
                              const int3 *indices,
                              int numIndices,
                              const float3 *vertices,
                              int numVertices,
                              const float3 *normals,
                              const float2 *texcoords)
  {
    LOG_API_ENTRY;
    BNGeom mesh = bnGeometryCreate(model,whichSlot,"triangles");
    
    BNData _vertices = bnDataCreate(model,whichSlot,BN_FLOAT3,numVertices,vertices);
    bnSetAndRelease(mesh,"vertices",_vertices);
    
    BNData _indices  = bnDataCreate(model,whichSlot,BN_INT3,numIndices,indices);
    bnSetAndRelease(mesh,"indices",_indices);
    
    if (normals) {
      BNData _normals  = bnDataCreate(model,whichSlot,BN_FLOAT3,normals?numVertices:0,normals);
      bnSetAndRelease(mesh,"normals",_normals);
    }
    
    if (texcoords) {
      BNData _texcoords  = bnDataCreate(model,whichSlot,BN_FLOAT2,texcoords?numVertices:0,texcoords);
      bnSetAndRelease(mesh,"texcoords",_texcoords);
    }
    bnAssignMaterial(mesh,material);
    bnCommit(mesh);
    return mesh;
  }  
  
  BN_API
  BNScalarField bnScalarFieldCreate(BNModel model,
                                    int whichSlot,
                                    const char *type)
  {
    ScalarField::SP sf = ScalarField::create(checkGet(model,whichSlot),type);
    if (!sf) return 0;
    return (BNScalarField)checkGet(model,whichSlot)->context->initReference(sf);
  }
  
  
  BN_API
  BNGeom bnGeometryCreate(BNModel model,
                          int whichSlot,
                          const char *type)
  {
    Geometry::SP geom = Geometry::create(checkGet(model,whichSlot),type);
    if (!geom) return 0;
    return (BNGeom)checkGet(model,whichSlot)->context->initReference(geom);
  }

  BN_API
  BNMaterial bnMaterialCreate(BNModel model,
                              int whichSlot,
                              const char *type)
  {
    render::HostMaterial::SP material
      = render::HostMaterial::create(checkGet(model,whichSlot),type);
    if (!material) return 0;
#if 0
    auto mat = checkGet(model,whichSlot)->context->initReference(material);
    material->commit();
    return (BNMaterial)mat;
#else
    return (BNMaterial)checkGet(model,whichSlot)->context->initReference(material);
#endif
  }

  /*! creates a cudaArray2D of specified size and texels. Can be passed
    to a sampler to create a matching cudaTexture2D */
  BN_API
  BNTextureData bnTextureData2DCreate(BNModel model,
                                      int whichSlot,
                                      BNTexelFormat texelFormat,
                                      int width, int height,
                                      const void *texels)
  {
    TextureData::SP data
      = std::make_shared<TextureData>(checkGet(model,whichSlot),
                                      texelFormat,vec3i(width,height,0),
                                      texels);
    if (!data) return 0;
    return (BNTextureData)checkGet(model,whichSlot)->context->initReference(data);
  }
  
  BN_API
  BNSampler bnSamplerCreate(BNModel model,
                            int whichSlot,
                            const char *type)
  {
    render::Sampler::SP sampler
      = render::Sampler::create(checkGet(model,whichSlot),type);
    if (!sampler) return 0;
    // sampler->commit();
    return (BNSampler)checkGet(model,whichSlot)->context->initReference(sampler);
  }

  BN_API
  BNCamera bnCameraCreate(BNContext context,
                          const char *type)
  {
    Camera::SP camera = Camera::create(checkGet(context),type);
    if (!camera) return 0;
    return (BNCamera)checkGet(context)->initReference(camera);
  }


  BN_API
  void bnVolumeSetXF(BNVolume volume,
                     float2 domain,
                     const float4 *_values,
                     int numValues,
                     float densityAt1)
  {
    LOG_API_ENTRY;
    std::vector<vec4f> values;
    assert(_values);

    for (int i=0;i<numValues;i++)
      values.push_back((const vec4f &)_values[i]);

    checkGet(volume)->setXF(range1f(domain.x,domain.y),values,densityAt1);
  }
  
  BN_API
  BNVolume bnVolumeCreate(BNModel model,
                          int whichSlot,
                          BNScalarField _sf)
  {
    ScalarField::SP sf = checkGetSP(_sf);
    return (BNVolume)checkGet(model,whichSlot)->createVolume
      (sf);
  }

  BN_API
  void bnSetRadiance(BNModel model,
                          int whichSlot,
                          float radiance)
  {
    checkGet(model,whichSlot)->setRadiance(radiance);
  }

  BN_API
  BNLight bnLightCreate(BNModel model,
                        int whichSlot,
                        const char *type)
  {
    return (BNLight)checkGet(model,whichSlot)->createLight(checkGet(type));
  }

  BN_API
  BNData bnDataCreate(BNModel model,
                      int whichSlot,
                      BNDataType dataType,
                      size_t numItems,
                      const void *items)
  {
    return (BNData)checkGet(model,whichSlot)->createData(dataType,numItems,items);
  }


  BN_API
  BNScalarField bnStructuredDataCreate(BNModel model,
                                       int whichSlot,
                                       int3 dims,
                                       BNScalarType type,
                                       const void *scalars,
                                       float3 gridOrigin,
                                       float3 gridSpacing)
  {
    BNTexelFormat texelFormat;
    switch (type) {
    case BN_SCALAR_FLOAT:
      texelFormat = BN_TEXEL_FORMAT_R32F;
      break;
    case BN_SCALAR_UINT8:
      texelFormat = BN_TEXEL_FORMAT_R8;
      break;
    default:
      throw std::runtime_error("unsupported structured data format #"+std::to_string((int)type));
    }
    
    BNScalarField sf
      = bnScalarFieldCreate(model,whichSlot,"structured");
    BNTexture3D texture
      = bnTexture3DCreate(model,whichSlot,texelFormat,
                          dims.x,dims.y,dims.z,scalars,BN_TEXTURE_LINEAR,BN_TEXTURE_CLAMP);
    bnSetObject(sf,"texture",texture);
    bnRelease(texture);
    bnSet3ic(sf,"dims",dims);
    bnSet3fc(sf,"gridOrigin",gridOrigin);
    bnSet3fc(sf,"gridSpacing",gridSpacing);
    bnCommit(sf);
    return sf;
  }
  
  BN_API
  BNScalarField bnUMeshCreate(BNModel model,
                              int whichSlot,
                              // vertices, 4 floats each (3 floats position,
                              // 4th float scalar value)
                              const float4  *vertices,
                              int            numVertices,
                              const int     *indices,
                              int            numIndices,
                              const int     *elementOffsets,
                              int            numElements,
                              // // tets, 4 ints in vtk-style each
                              // const int *_tetIndices, int numTets,
                              // // pyramids, 5 ints in vtk-style each
                              // const int *_pyrIndices, int numPyrs,
                              // // wedges/tents, 6 ints in vtk-style each
                              // const int *_wedIndices, int numWeds,
                              // // general (non-guaranteed cube/voxel) hexes, 8
                              // // ints in vtk-style each
                              // const int *_hexIndices, int numHexes,
                              // //
                              // int numGrids,
                              // // offsets into gridScalars array
                              // const int *_gridOffsets,
                              // // grid dims (3 floats each)
                              // const int *_gridDims,
                              // // grid domains, 6 floats each (3 floats min corner,
                              // // 3 floats max corner)
                              // const float *_gridDomains,
                              // // grid scalars
                              // const float *_gridScalars,
                              // int numGridScalars,
                              const float3 *domainOrNull    
                              )
  {
    std::cout << "#bn: copying umesh from app ..." << std::endl;
    // std::vector<vec4f>      vertices(numVertices);
    // std::vector<vec4f>      vertices(numVertices);
    // std::vector<int>        gridOffsets(numGrids);
    // std::vector<vec3i>      gridDims(numGrids);
    // std::vector<box4f>      gridDomains(numGrids);
    // std::vector<float>      gridScalars(numGridScalars);
    // std::vector<TetIndices> tetIndices(numTets);
    // std::vector<PyrIndices> pyrIndices(numPyrs);
    // std::vector<WedIndices> wedIndices(numWeds);
    // std::vector<HexIndices> hexIndices(numHexes);
    // memcpy(tetIndices.data(),_tetIndices,tetIndices.size()*sizeof(tetIndices[0]));
    // memcpy(pyrIndices.data(),_pyrIndices,pyrIndices.size()*sizeof(pyrIndices[0]));
    // memcpy(wedIndices.data(),_wedIndices,wedIndices.size()*sizeof(wedIndices[0]));
    // memcpy(hexIndices.data(),_hexIndices,hexIndices.size()*sizeof(hexIndices[0]));
    // memcpy(vertices.data(),_vertices,vertices.size()*sizeof(vertices[0]));
    // memcpy(gridOffsets.data(),_gridOffsets,gridOffsets.size()*sizeof(gridOffsets[0]));
    // memcpy(gridDims.data(),_gridDims,gridDims.size()*sizeof(gridDims[0]));
    // memcpy(gridDomains.data(),_gridDomains,gridDomains.size()*sizeof(gridDomains[0]));
    // memcpy(gridScalars.data(),_gridScalars,gridScalars.size()*sizeof(gridScalars[0]));

    box3f domain
      = domainOrNull
      ? *(const box3f*)domainOrNull
      : box3f();
    
    ScalarField::SP sf = 
      UMeshField::create(checkGet(model,whichSlot),
                         (const vec4f*)vertices,numVertices,
                         indices,numIndices,
                         elementOffsets,
                         numElements,
                         // tetIndices,
                         // pyrIndices,
                         // wedIndices,
                         // hexIndices,
                         // gridOffsets,
                         // gridDims,
                         // gridDomains,
                         // gridScalars,
                         domain);
    return (BNScalarField)checkGet(model,whichSlot)->context->initReference(sf);
  }
  
  BN_API
  BNScalarField bnBlockStructuredAMRCreate(BNModel model,
                                           int whichSlot,
                                           /*TODO:const float *cellWidths,*/
                                           // block bounds, 6 ints each (3 for min,
                                           // 3 for max corner)
                                           const int *_blockBounds, int numBlocks,
                                           // refinement level, per block,
                                           // finest is level 0,
                                           const int *_blockLevels,
                                           // offsets into blockData array
                                           const int *_blockOffsets,
                                           // block scalars
                                           const float *_blockScalars,
                                           int numBlockScalars)
  {
    std::cout << "#bn: copying 'amr' from app ..." << std::endl;
    std::vector<box3i> blockBounds(numBlocks);
    std::vector<int>   blockLevels(numBlocks);
    std::vector<int>   blockOffsets(numBlocks);
    std::vector<float> blockScalars(numBlockScalars);
    memcpy(blockBounds.data(),_blockBounds,blockBounds.size()*sizeof(blockBounds[0]));
    memcpy(blockLevels.data(),_blockLevels,blockLevels.size()*sizeof(blockLevels[0]));
    memcpy(blockOffsets.data(),_blockOffsets,blockOffsets.size()*sizeof(blockOffsets[0]));
    memcpy(blockScalars.data(),_blockScalars,blockScalars.size()*sizeof(blockScalars[0]));
    ScalarField *sf = checkGet(model,whichSlot)
      ->createBlockStructuredAMR(blockBounds,
                                 blockLevels,
                                 blockOffsets,
                                 blockScalars);
    return (BNScalarField)sf;
  }


  BN_API
  BNGroup bnGroupCreate(BNModel model,
                        int whichSlot,
                        BNGeom *geoms, int numGeoms,
                        BNVolume *volumes, int numVolumes)
  {
    LOG_API_ENTRY;
    std::vector<Geometry::SP> _geoms;
    for (int i=0;i<numGeoms;i++)
      _geoms.push_back(checkGet(geoms[i])->as<Geometry>());
    std::vector<Volume::SP> _volumes;
    for (int i=0;i<numVolumes;i++)
      _volumes.push_back(checkGet(volumes[i])->as<Volume>());
    Group *group = checkGet(model,whichSlot)->createGroup(_geoms,_volumes);
    return (BNGroup)group;
  }

  BN_API
  void  bnGroupBuild(BNGroup group)
  {
    LOG_API_ENTRY;
    checkGet(group)->build();
  }
  
  BN_API
  void  bnBuild(BNModel model,
                int whichSlot)
  {
    LOG_API_ENTRY;
    checkGet(model,whichSlot)->build();
  }
  
  BN_API
  void bnCommit(BNObject target)
  {
    checkGet(target)->commit();
  }
  
              
  BN_API
  void bnSetString(BNObject target, const char *param, const char *value)
  {
    if (!checkGet(target)->setString(checkGet(param),value))
      checkGet(target)->warn_unsupported_member(param,"std::string");
  }

  BN_API
  void bnSetData(BNObject target, const char *param, BNData value)
  {
    if (!checkGet(target)->setData(checkGet(param),
                                   checkGet(value)->shared_from_this()->as<Data>()))
      checkGet(target)->warn_unsupported_member(param,"BNData");
  }

  BN_API
  void bnSetObject(BNObject target, const char *param, BNObject value)
  {
    Object::SP asObject
      = value 
      ? checkGet(value)->shared_from_this()
      : Object::SP{};
    bool accepted = checkGet(target)->setObject(checkGet(param),asObject);
    if (!accepted)
      checkGet(target)->warn_unsupported_member(param,"BNObject");
  }

  BN_API
  void bnSet1i(BNObject target, const char *param, int x)
  {
    if (!checkGet(target)->set1i(checkGet(param),x))
      checkGet(target)->warn_unsupported_member(param,"int");
  }

  BN_API
  void bnSet2i(BNObject target, const char *param, int x, int y)
  {
    if (!checkGet(target)->set2i(checkGet(param),vec2i(x,y)))
      checkGet(target)->warn_unsupported_member(param,"vec2i");
  }

  BN_API
  void bnSet3i(BNObject target, const char *param, int x, int y, int z)
  {
    if (!checkGet(target)->set3i(checkGet(param),vec3i(x,y,z)))
      checkGet(target)->warn_unsupported_member(param,"vec3i");
  }

  BN_API
  void bnSet3ic(BNObject target, const char *param, int3 value)
  {
    if (!checkGet(target)->set3i(checkGet(param),(const vec3i&)value))
      checkGet(target)->warn_unsupported_member(param,"vec3i");
  }

  BN_API
  void bnSet4i(BNObject target, const char *param, int x, int y, int z, int w)
  {
    if (!checkGet(target)->set4i(checkGet(param),vec4i(x,y,z,w)))
      checkGet(target)->warn_unsupported_member(param,"vec4i");
  }

  BN_API
  void bnSet1f(BNObject target, const char *param, float value)
  {
    if (!checkGet(target)->set1f(checkGet(param),value))
      checkGet(target)->warn_unsupported_member(param,"float");
  }

  BN_API
  void bnSet3f(BNObject target, const char *param, float x, float y, float z)
  {
    if (!checkGet(target)->set3f(checkGet(param),vec3f(x,y,z)))
      checkGet(target)->warn_unsupported_member(param,"vec3f");
  }

  BN_API
  void bnSet4f(BNObject target, const char *param, float x, float y, float z, float w)
  {
    if (!checkGet(target)->set4f(checkGet(param),vec4f(x,y,z,w)))
      checkGet(target)->warn_unsupported_member(param,"vec4f");
  }

  BN_API
  void bnSet3fc(BNObject target, const char *param, float3 value)
  {
    if (!checkGet(target)->set3f(checkGet(param),(const vec3f&)value))
      checkGet(target)->warn_unsupported_member(param,"vec3f");
  }

  BN_API
  void bnSet4x3fv(BNObject target, const char *param, const float *transform)
  {
    assert(transform);
    if (!checkGet(target)->set4x3f(checkGet(param),*(const affine3f*)transform))
      checkGet(target)->warn_unsupported_member(param,"affine3f");
  }

  BN_API
  void bnSet4x4fv(BNObject target, const char *param, const float *transform)
  {
    assert(transform);
    if (!checkGet(target)->set4x4f(checkGet(param),*(const mat4f*)transform))
      checkGet(target)->warn_unsupported_member(param,"mat4f");
  }
  



  

  
  BN_API
  BNFrameBuffer bnFrameBufferCreate(BNContext context,
                                    int owningRank)
  {
    LOG_API_ENTRY;
    FrameBuffer *fb = checkGet(context)->createFB(owningRank);
    return (BNFrameBuffer)fb;
  }

  BN_API
  void bnFrameBufferResize(BNFrameBuffer fb,
                           int sizeX, int sizeY,
                           uint32_t *hostRGBA,
                           float *hostDepth)
  {
    LOG_API_ENTRY;
    checkGet(fb)->resize(vec2i{sizeX,sizeY},hostRGBA,hostDepth);
  }


  BN_API
  void bnAccumReset(BNFrameBuffer fb)
  {
    checkGet(fb)->resetAccumulation();
  }
  
  BN_API
  void bnRender(BNModel model,
                BNCamera camera,
                BNFrameBuffer fb,
                int pathsPerPixel)
  {
    // static double t_first = getCurrentTime();
    // static double t_sum = 0.;
    
    // double t0 = getCurrentTime();
    // LOG_API_ENTRY;
    checkGet(model)->render(checkGet(camera),checkGet(fb),pathsPerPixel);
    // double t1 = getCurrentTime();

    // t_sum += (t1-t0);
    // printf("time in %f\n",float((t_sum / (t1 - t_first))));
  }

  BN_API
  BNContext bnContextCreate(/*! how many data slots this context is to
                              offer, and which part(s) of the
                              distributed model data these slot(s)
                              will hold */
                            const int *dataRanksOnThisContext,
                            int        numDataRanksOnThisContext,
                            /*! which gpu(s) to use for this
                              process. default is to distribute
                              node's GPUs equally over all ranks on
                              that given node */
                            const int *_gpuIDs,
                            int  numGPUs)
  {
    LOG_API_ENTRY;
    // ------------------------------------------------------------------
    // create vector of data groups; if actual specified by user we
    // use those; otherwise we use IDs
    // [0,1,...numModelSlotsOnThisHost)
    // ------------------------------------------------------------------
    assert(numDataRanksOnThisContext > 0);
    std::vector<int> dataGroupIDs;
    for (int i=0;i<numDataRanksOnThisContext;i++)
      dataGroupIDs.push_back
        (dataRanksOnThisContext
         ? dataRanksOnThisContext[i]
         : i);

    // ------------------------------------------------------------------
    // create list of GPUs to use for this rank. if specified by user
    // we use this; otherwise we use GPUs in order, split into groups
    // according to how many ranks there are on this host. Ie, if host
    // has four GPUs the first rank will take 0 and 1; and the second
    // one will take 2 and 3.
    // ------------------------------------------------------------------
    std::vector<int> gpuIDs;
    if (_gpuIDs) {
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back(_gpuIDs[i]);
    } else {
      if (numGPUs < 1)
        cudaGetDeviceCount(&numGPUs);
      for (int i=0;i<numGPUs;i++)
        gpuIDs.push_back(i);
    }
    if (gpuIDs.empty())
      throw std::runtime_error("no GPUs!?");

    if (gpuIDs.size() < numDataRanksOnThisContext) {
      std::vector<int> replicatedIDs;
      for (int i=0;i<numDataRanksOnThisContext;i++)
        replicatedIDs.push_back(gpuIDs[i % gpuIDs.size()]);
      gpuIDs = replicatedIDs;
    }
    
    return (BNContext)new LocalContext(dataGroupIDs,
                                       gpuIDs);
  }

}
