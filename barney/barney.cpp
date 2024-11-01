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
#include "barney/amr//BlockStructuredField.h"
#include "barney/common/Data.h"
#include "barney/common/mat4.h"
#include "barney/Camera.h"

#define WARN_NOTIMPLEMENTED std::cout << " ## " << __PRETTY_FUNCTION__ << " not implemented yet ..." << std::endl;

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif

#ifdef NDEBUG
# define BARNEY_ENTER(fct) /* nothing */
# define BARNEY_LEAVE(fct,retValue) /* nothing */
#else
# define BARNEY_ENTER(fct) try {                \
    if (0) std::cout << "@bn.entering " << fct << std::endl;     \



# define BARNEY_LEAVE(fct,retValue)                                     \
  } catch (std::exception e) {                                           \
    std::cerr << OWL_TERMINAL_RED << "@" << fct << ": "              \
              << e.what() << OWL_TERMINAL_DEFAULT << std::endl;      \
    return retValue ;                                                   \
  }
#endif

namespace barney {

  inline Context *checkGet(BNContext context)
  {
    assert(context);
    return (Context *)context;
  }
  
  inline Renderer *checkGet(BNRenderer renderer)
  {
    assert(renderer);
    return (Renderer *)renderer;
  }
  
  inline std::string checkGet(const char *s)
  {
    assert(s != nullptr);
    return s;
  }
  
  inline Object *checkGet(BNObject object)
  {
    if (!object) throw std::runtime_error
                   ("@barney: trying to use/access null object");
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
  
  inline ModelSlot *checkGet(BNModel model, int slot)
  {
    return checkGet(model)->getSlot(slot);
  }
  
  inline Geometry *checkGet(BNGeom geom)
  {
    if (!geom) throw std::runtime_error("@barney: trying to use/access null geometry");
    assert(geom);
    return (Geometry *)geom;
  }
  
  inline Volume *checkGet(BNVolume volume)
  {
    if (!volume) throw std::runtime_error("@barney: trying to use/access null volume");
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
    if (!group) throw std::runtime_error
                   ("@barney: trying to use/access null group");
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

  /*! creates a cudaArray2D of specified size and texels. Can be passed
    to a sampler to create a matching cudaTexture2D, or as a background
    image to a renderer */
  BN_API
  BNTextureData bnTextureData2DCreate(BNContext context,
                                      int slot,
                                      BNTexelFormat texelFormat,
                                      int width, int height,
                                      const void *texels)
  {
    LOG_API_ENTRY;
    TextureData::SP data
      = std::make_shared<TextureData>(checkGet(context),slot,
                                      texelFormat,vec3i(width,height,0),
                                      texels);
    if (!data) return 0;
    return (BNTextureData)checkGet(context)->initReference(data);
  }

  // // ------------------------------------------------------------------
  // /*! creates a cudaArray2D of specified size and texels. Can be passed
  //   to a sampler to create a matching cudaTexture2D */
  // BN_API
  // BNTextureData bnTextureData2DCreate(BNModel model,
  //                                     int slot,
  //                                     BNTexelFormat texelFormat,
  //                                     int width, int height,
  //                                     const void *texels)
  // {
  //   LOG_API_ENTRY;
  //   Context *context = checkGetContext(model);
  //   TextureData::SP data
  //     = std::make_shared<TextureData>(context,slot,
  //                                     texelFormat,vec3i(width,height,0),
  //                                     texels);
  //   if (!data) return 0;
  //   return (BNTextureData)context->initReference(data);
  // }
  
  
  
  // ------------------------------------------------------------------
  BN_API
  BNTexture2D bnTexture2DCreate(BNContext context,
                                int slot,
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
    assert(context);
    Texture::SP tex
      = std::make_shared<Texture>(checkGet(context),slot,
                                  texelFormat,vec2i(size_x,size_y),texels,
                                  filterMode,addressMode,colorSpace);
    return (BNTexture)checkGet(context)->initReference(tex);
  }

  BN_API
  BNTexture3D bnTexture3DCreate(BNContext context,
                                int slot,
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
      (checkGet(context),slot,
       texelFormat,vec3i(size_x,size_y,size_z),texels,
       filterMode,addressMode);
    ;
    return (BNTexture3D)checkGet(context)->initReference(tex);
  }
  
  // ------------------------------------------------------------------
  
  BN_API
  BNModel bnModelCreate(BNContext ctx)
  {
    LOG_API_ENTRY;
    return (BNModel)checkGet(ctx)->createModel();
  }

  BN_API
  BNRenderer bnRendererCreate(BNContext ctx, const char *ignoreForNow)
  {
    LOG_API_ENTRY;
    return (BNRenderer)checkGet(ctx)->createRenderer();
  }

  BN_API
  void bnSetInstances(BNModel model,
                      int slot,
                      BNGroup *_groups,
                      BNTransform *xfms,
                      int numInstances)
  {
    LOG_API_ENTRY;

    std::vector<Group::SP> groups;
    for (int i=0;i<numInstances;i++) {
      groups.push_back(checkGetSP(_groups[i]));
    }
    checkGet(model,slot)->setInstances(groups,(const affine3f *)xfms);
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
  BNGeom bnTriangleMeshCreate(BNContext context,
                              int slot,
                              const BNMaterialHelper *material,
                              const int3 *indices,
                              int numIndices,
                              const float3 *vertices,
                              int numVertices,
                              const float3 *normals,
                              const float2 *texcoords)
  {
    LOG_API_ENTRY;
    BNGeom mesh = bnGeometryCreate(context,slot,"triangles");
    
    BNData _vertices = bnDataCreate(context,slot,BN_FLOAT3,numVertices,vertices);
    bnSetAndRelease(mesh,"vertices",_vertices);
    
    BNData _indices  = bnDataCreate(context,slot,BN_INT3,numIndices,indices);
    bnSetAndRelease(mesh,"indices",_indices);
    
    if (normals) {
      BNData _normals  = bnDataCreate(context,slot,BN_FLOAT3,normals?numVertices:0,normals);
      bnSetAndRelease(mesh,"normals",_normals);
    }
    
    if (texcoords) {
      BNData _texcoords  = bnDataCreate(context,slot,BN_FLOAT2,texcoords?numVertices:0,texcoords);
      bnSetAndRelease(mesh,"texcoords",_texcoords);
    }
    bnAssignMaterial(mesh,material);
    bnCommit(mesh);
    return mesh;
  }  
  
  BN_API
  BNScalarField bnScalarFieldCreate(BNContext context,
                                    int slot,
                                    const char *type)
  {
    ScalarField::SP sf = ScalarField::create(checkGet(context),slot,type);
    return (BNScalarField)checkGet(context)->initReference(sf);
  }
  
  
  BN_API
  BNGeom bnGeometryCreate(BNContext context,
                          int slot,
                          const char *type)
  {
    Geometry::SP geom = Geometry::create(checkGet(context),slot,type);
    return (BNGeom)checkGet(context)->initReference(geom);
  }

  BN_API
  BNMaterial bnMaterialCreate(BNContext context,
                              int slot,
                              const char *type)
  {
    render::HostMaterial::SP material
      = render::HostMaterial::create(checkGet(context),slot,type);
    return (BNMaterial)checkGet(context)->initReference(material);
  }

  BN_API
  BNSampler bnSamplerCreate(BNContext context,
                            int slot,
                            const char *type)
  {
    render::Sampler::SP sampler
      = render::Sampler::create(checkGet(context),slot,type);
    if (!sampler) return 0;
    // sampler->commit();
    return (BNSampler)checkGet(context)->initReference(sampler);
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
  BNVolume bnVolumeCreate(BNContext context,
                          int slot,
                          BNScalarField _sf)
  {
    LOG_API_ENTRY;
    Volume::SP volume = Volume::create(checkGetSP(_sf));
    return (BNVolume)checkGet(context)->initReference(volume);
  }

  BN_API
  BNLight bnLightCreate(BNContext context,
                        int slot,
                        const char *type)
  {
    LOG_API_ENTRY;
    Light::SP light = Light::create(checkGet(context),slot,type);
    return (BNLight)checkGet(context)->initReference(light);
  }

  BN_API
  BNData bnDataCreate(BNContext context,
                      int slot,
                      BNDataType dataType,
                      size_t numItems,
                      const void *items)
  {
    LOG_API_ENTRY;
    Data::SP data = Data::create(checkGet(context),slot,dataType,numItems,items);
    return (BNData)checkGet(context)->initReference(data);
  }


  BN_API
  BNScalarField bnStructuredDataCreate(BNContext context,
                                       int slot,
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
      = bnScalarFieldCreate(context,slot,"structured");
    BNTexture3D texture
      = bnTexture3DCreate(context,slot,texelFormat,
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
  BNScalarField bnUMeshCreate(BNContext context,
                              int slot,
                              // vertices, 4 floats each (3 floats position,
                              // 4th float scalar value)
                              const float4  *vertices,
                              int            numVertices,
                              const int     *indices,
                              int            numIndices,
                              const int     *elementOffsets,
                              int            numElements,
                              const float3 *domainOrNull    
                              )
  {
    box3f domain
      = domainOrNull
      ? *(const box3f*)domainOrNull
      : box3f();
    
    ScalarField::SP sf = 
      UMeshField::create(checkGet(context),slot,
                         (const vec4f*)vertices,numVertices,
                         indices,numIndices,
                         elementOffsets,
                         numElements,
                         domain);
    return (BNScalarField)checkGet(context)->initReference(sf);
  }
  
  BN_API
  BNScalarField bnBlockStructuredAMRCreate(BNContext context,
                                           int slot,
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
    ScalarField::SP sf =
      std::make_shared<BlockStructuredField>(checkGet(context),slot,
                                             blockBounds,
                                             blockLevels,
                                             blockOffsets,
                                             blockScalars);
    return (BNScalarField)checkGet(context)->initReference(sf);
  }


  BN_API
  BNGroup bnGroupCreate(BNContext context,
                        int slot,
                        BNGeom *geoms, int numGeoms,
                        BNVolume *volumes, int numVolumes)
  {
    LOG_API_ENTRY;
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    // try {
    std::vector<Geometry::SP> _geoms;
    for (int i=0;i<numGeoms;i++)
      _geoms.push_back(checkGet(geoms[i])->as<Geometry>());
    std::vector<Volume::SP> _volumes;
    for (int i=0;i<numVolumes;i++)
      _volumes.push_back(checkGet(volumes[i])->as<Volume>());
    Group::SP group
      = std::make_shared<Group>(checkGet(context),slot,
                                _geoms,_volumes);
    return (BNGroup)checkGet(context)->initReference(group);
    // } catch (std::runtime_error error) {
    //   std::cerr << "@barney: Error in creating group: " << error.what()
    //             << "... going to return null group." << std::endl;
    //   return BNGroup{0};
    // }
    BARNEY_LEAVE(__PRETTY_FUNCTION__,0);
  }

  BN_API
  void  bnGroupBuild(BNGroup group)
  {
    LOG_API_ENTRY;
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    if (!group) {
      std::cerr << "@barney(WARNING): bnGroupBuild with null group - ignoring this, but this is is an app error that should be fixed, and is only likely to cause issues later on" << std::endl;
      return;
    }
    checkGet(group)->build();
    BARNEY_LEAVE(__PRETTY_FUNCTION__,);
  }
  
  BN_API
  void  bnBuild(BNModel model,
                int slot)
  {
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    LOG_API_ENTRY;
    checkGet(model,slot)->build();
    BARNEY_LEAVE(__PRETTY_FUNCTION__,);
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
  void bnSet4fc(BNObject target, const char *param, float4 value)
  {
    if (!checkGet(target)->set4f(checkGet(param),(const vec4f&)value))
      checkGet(target)->warn_unsupported_member(param,"vec4f");
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
  void bnRender(BNRenderer renderer,
                BNModel    model,
                BNCamera   camera,
                BNFrameBuffer fb)
  {
    // static double t_first = getCurrentTime();
    // static double t_sum = 0.;
    
    // double t0 = getCurrentTime();
    // LOG_API_ENTRY;
    checkGet(model)->render(checkGet(renderer),checkGet(camera),checkGet(fb));
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
