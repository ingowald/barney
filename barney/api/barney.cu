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

#include "barney/api/Context.h"
#if BARNEY_MPI
# include "barney/common/MPIWrappers.h"
# include "barney/barney_mpi.h"
#endif

static_assert(sizeof(size_t) == 8, "Trying to compile in 32-bit mode ... this isn't going to work");

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
# define BARNEY_ENTER(fct) try {                                 \
  if (0) std::cout << "@bn.entering " << fct << std::endl;       \
  


# define BARNEY_LEAVE(fct,retValue)                                     \
  } catch (std::exception &e) {                                         \
    std::cerr << OWL_TERMINAL_RED << "@" << fct << ": "                 \
              << e.what() << OWL_TERMINAL_DEFAULT << std::endl;         \
    return retValue ;                                                   \
  }
#endif

namespace barney_api {

  extern "C" {
#if BARNEY_BACKEND_EMBREE
    barney_api::Context *
    createContext_embree(const std::vector<int> &dgIDs);
#endif
#if BARNEY_BACKEND_OPTIX
    barney_api::Context *
    createContext_optix(const std::vector<int> &dgIDs,
                        int numGPUs, const int *gpuIDs);
#endif
#if BARNEY_MPI
# if BARNEY_BACKEND_EMBREE
    barney_api::Context *
    createMPIContext_embree(barney_api::mpi::Comm world,
                            barney_api::mpi::Comm workers,
                            bool isActiveWorker,
                            const std::vector<int> &dgIDs);
# endif
# if BARNEY_BACKEND_OPTIX
    barney_api::Context *
    createMPIContext_optix(barney_api::mpi::Comm world,
                           barney_api::mpi::Comm workers,
                           bool isActiveWorker,
                           const std::vector<int> &dgIDs,
                           const std::vector<int> &gpuIDs);
# endif
#endif
  }
  
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
  
  inline Model *checkGet(BNModel model)
  {
    assert(model);
    return (Model *)model;
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
  
  // inline GlobalModel *checkGet(BNModel model)
  // {
  //   assert(model);
  //   return (GlobalModel *)model;
  // }
  
  // inline ModelSlot *checkGet(BNModel model, int slot)
  // {
  //   return checkGet(model)->getSlot(slot);
  // }
  
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
  inline std::shared_ptr<Group> checkGetSP(BNGroup group)
  {
    return checkGet(group)->shared_from_this()->as<Group>();
  }
  
  // ------------------------------------------------------------------
  inline std::shared_ptr<Data> checkGetSP(BNData data)
  {
    return checkGet(data)->shared_from_this()->as<Data>();
  }
  
  inline std::shared_ptr<ScalarField> checkGetSP(BNScalarField sf)
  {
    return checkGet(sf)->shared_from_this()->as<ScalarField>();
  }
  // ------------------------------------------------------------------

  /*! creates a cudaArray2D of specified size and texels. Can be passed
    to a sampler to create a matching cudaTexture2D, or as a background
    image to a renderer */
  BARNEY_API
  BNTextureData bnTextureData2DCreate(BNContext _context,
                                      int slot,
                                      BNDataType texelFormat,
                                      int width, int height,
                                      const void *texels)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,
                                     texelFormat,
                                     vec3i(width,height,0),
                                     texels);
    return (BNTextureData)context->initReference(td);
  }
  
  BARNEY_API
  BNTextureData bnTextureData3DCreate(BNContext _context,
                                      int slot,
                                      BNDataType texelFormat,
                                      int width, int height, int depth,
                                      const void *texels)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,
                                   texelFormat,
                                   vec3i(width,height,depth),
                                   texels);
    return (BNTextureData)context->initReference(td);
  }

  
  // ------------------------------------------------------------------
  BARNEY_API
  BNTexture2D bnTexture2DCreate(BNContext _context,
                                int slot,
                                BNDataType texelFormat,
                                /*! number of texels in x dimension */
                                uint32_t size_x,
                                /*! number of texels in y dimension */
                                uint32_t size_y,
                                const void *texels,
                                BNTextureFilterMode  filterMode,
                                BNTextureAddressMode addressMode_x,
                                BNTextureAddressMode addressMode_y,
                                BNTextureColorSpace  colorSpace)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
#if 1
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,texelFormat,
                                   vec3i(size_x,size_y,0),texels);
    // BNTextureData td
    //   = bnTextureData2DCreate(_context,slot,texelFormat,
    //                           size_x,size_y,texels);
    BNTextureAddressMode addressModes[3] = {
      addressMode_x,addressMode_y,(BNTextureAddressMode)0
    };
    std::shared_ptr<Texture> tex
      = context->createTexture(td,
                               filterMode,
                               addressModes,
                               colorSpace);
    return (BNTexture2D)context->initReference(tex);
#else
    auto devices = context->getDevices(slot);
    TextureData::SP td
      = std::make_shared<TextureData>(context,devices,
                                      texelFormat,
                                      vec3i(size_x,size_y,0),
                                      texels);
    Texture::SP tex
      = std::make_shared<Texture>(context,devices,
                                  td,filterMode,
                                  addressMode_x,addressMode_y,
                                  colorSpace);
    
    return (BNTexture)context->initReference(tex);
#endif
  }

  BARNEY_API
  BNTexture3D bnTexture3DCreate(BNContext _context,
                                int slot,
                                BNDataType texelFormat,
                                uint32_t size_x,
                                uint32_t size_y,
                                uint32_t size_z,
                                const void *texels,
                                BNTextureFilterMode  filterMode,
                                BNTextureAddressMode addressMode)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
#if 1
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,texelFormat,
                                   vec3i(size_x,size_y,size_z),texels);
    BNTextureAddressMode addressModes[3] = {
      addressMode,addressMode,addressMode
    };
    std::shared_ptr<Texture> tex
      = context->createTexture(td,
                               filterMode,
                               addressModes,
                               BN_COLOR_SPACE_LINEAR);
    return (BNTexture3D)context->initReference(tex);
#else
    auto devices = context->getDevices(slot);
    TextureData::SP td
      = std::make_shared<TextureData>(context,devices,
                                      texelFormat,
                                      vec3i(size_x,size_y,size_z),
                                      texels);
    Texture3D::SP tex
      = std::make_shared<Texture3D>(context,devices,
                                    td,filterMode,addressMode);
    
    return (BNTexture3D)context->initReference(tex);
#endif
  }
  
  // ------------------------------------------------------------------
  
  BARNEY_API
  BNModel bnModelCreate(BNContext _context)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Model> model = context->createModel();
    return (BNModel)context->initReference(model);
  }

  BARNEY_API
  BNRenderer bnRendererCreate(BNContext _context,
                              const char *ignoreForNow)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Renderer> renderer = context->createRenderer();
    return (BNRenderer)context->initReference(renderer);
  }

  BARNEY_API
  void bnSetInstances(BNModel model,
                      int slot,
                      BNGroup *_groups,
                      BNTransform *xfms,
                      int numInstances)
  {
    LOG_API_ENTRY;
    checkGet(model)->setInstances(slot,
                                  (Group **)_groups,
                                  (const affine3f *)xfms,
                                  numInstances);
    // std::vector<Group::SP> groups;
    // for (int i=0;i<numInstances;i++) {
    //   groups.push_back(checkGetSP(_groups[i]));
    // }
    // checkGet(model,slot)->setInstances(groups,(const affine3f *)xfms);
  }
  
  BARNEY_API
  void  bnRelease(BNObject _object)
  {
    LOG_API_ENTRY;
    Object *object = checkGet(_object);
    assert(object);
    Context *context = object->getContext();
    assert(context);
    context->releaseHostReference(object->shared_from_this());
  }
  
  BARNEY_API
  void  bnAddReference(BNObject _object)
  {
    LOG_API_ENTRY;
    if (_object == 0) return;
    Object *object = checkGet(_object);
    Context *context = object->getContext();
    context->addHostReference(object->shared_from_this());
  }


  BARNEY_API
  void bnContextDestroy(BNContext context)
  {
    LOG_API_ENTRY;
    delete (Context *)context;
  }

#if 0
  BARNEY_API
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
#endif
  BARNEY_API
  BNScalarField bnScalarFieldCreate(BNContext _context,
                                    int slot,
                                    const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<ScalarField> sf
      = context->createScalarField(slot,type);
    return (BNScalarField)context->initReference(sf);
  }
  
  
  BARNEY_API
  BNGeom bnGeometryCreate(BNContext _context,
                          int slot,
                          const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<Geometry> geom
      = context->createGeometry(slot,type);
    return (BNGeom)context->initReference(geom);
  }

  // BARNEY_API
  // void bnCountAvailableDevice(int *numGPUs)
  // {
  //   if (!numGPUs) return;
  //   *numGPUs = rtc::Backend::getDeviceCount();
  // }

  
  BARNEY_API
  BNMaterial bnMaterialCreate(BNContext _context,
                              int slot,
                              const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<Material> material
      = context->createMaterial(slot,type);
    return (BNMaterial)context->initReference(material);
  }

  BARNEY_API
  BNSampler bnSamplerCreate(BNContext _context,
                            int slot,
                            const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<Sampler> sampler
      = context->createSampler(slot,type);
    if (!sampler) return 0;
    return (BNSampler)context->initReference(sampler);
  }
  
  BARNEY_API
  BNCamera bnCameraCreate(BNContext _context,
                          const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<Camera> camera
      = context->createCamera(type);
    if (!camera) return 0;
    return (BNCamera)context->initReference(camera);
  }


  BARNEY_API
  void bnVolumeSetXF(BNVolume volume,
                     bn_float2 domain,
                     const bn_float4 *_values,
                     int numValues,
                     float densityAt1)
  {
    LOG_API_ENTRY;
    assert(_values);

    checkGet(volume)->setXF(range1f(domain.x,domain.y),
                            _values,numValues,
                            densityAt1);
  }
  
  BARNEY_API
  BNVolume bnVolumeCreate(BNContext _context,
                          int slot,
                          BNScalarField _sf)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    
    std::shared_ptr<ScalarField> sf = checkGetSP(_sf);
    std::shared_ptr<Volume> volume
      = context->createVolume(checkGetSP(_sf));
    return (BNVolume)context->initReference(volume);
  }

  BARNEY_API
  BNLight bnLightCreate(BNContext _context,
                        int slot,
                        const char *type)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Light> light
      = context->createLight(slot,type);
    return (BNLight)context->initReference(light);
  }

  BARNEY_API
  BNData bnDataCreate(BNContext _context,
                      int slot,
                      BNDataType dataType,
                      size_t numItems,
                      const void *items)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Data> data
      = context->createData(slot,dataType,numItems,items);
    return (BNData)context->initReference(data);
  }


#if 0
  BARNEY_API
  BNScalarField bnStructuredDataCreate(BNContext context,
                                       int slot,
                                       int3 dims,
                                       BNDataType type,
                                       const void *scalars,
                                       float3 gridOrigin,
                                       float3 gridSpacing)
  {
    BNScalarField sf
      = bnScalarFieldCreate(context,slot,"structured");
    BNTexture3D texture
      = bnTexture3DCreate(context,slot,type,
                          dims.x,dims.y,dims.z,scalars,
                          BN_TEXTURE_LINEAR,BN_TEXTURE_CLAMP);
    bnSetObject(sf,"texture",texture);
    bnRelease(texture);
    bnSet3ic(sf,"dims",dims);
    bnSet3fc(sf,"gridOrigin",gridOrigin);
    bnSet3fc(sf,"gridSpacing",gridSpacing);
    bnCommit(sf);
    return sf;
  }
#endif
  
//   BARNEY_API
//   BNScalarField bnUMeshCreate(BNContext context,
//                               int slot,
//                               // vertices, 4 floats each (3 floats position,
//                               // 4th float scalar value)
//                               const bn_float4  *vertices,
//                               int            numVertices,
//                               const int     *indices,
//                               int            numIndices,
//                               const int     *elementOffsets,
//                               int            numElements,
//                               const float3 *domainOrNull    
//                               )
//   {
//     box3f domain
//       = domainOrNull
//       ? *(const box3f*)domainOrNull
//       : box3f();

// #if 1
//     BARNEY_NYI();
// #else
//     ScalarField::SP sf = 
//       UMeshField::create(checkGet(context),slot,
//                          (const vec4f*)vertices,numVertices,
//                          indices,numIndices,
//                          elementOffsets,
//                          numElements,
//                          domain);
//     return (BNScalarField)checkGet(context)->initReference(sf);
// #endif
//   }
  
//   BARNEY_API
//   BNScalarField bnBlockStructuredAMRCreate(BNContext context,
//                                            int slot,
//                                            /*TODO:const float *cellWidths,*/
//                                            // block bounds, 6 ints each (3 for min,
//                                            // 3 for max corner)
//                                            const int *_blockBounds, int numBlocks,
//                                            // refinement level, per block,
//                                            // finest is level 0,
//                                            const int *_blockLevels,
//                                            // offsets into blockData array
//                                            const int *_blockOffsets,
//                                            // block scalars
//                                            const float *_blockScalars,
//                                            int numBlockScalars)
//   {
// #if 1
//     BARNEY_NYI();
// #else
//     std::cout << "#bn: copying 'amr' from app ..." << std::endl;
//     std::vector<box3i> blockBounds(numBlocks);
//     std::vector<int>   blockLevels(numBlocks);
//     std::vector<int>   blockOffsets(numBlocks);
//     std::vector<float> blockScalars(numBlockScalars);
//     memcpy(blockBounds.data(),_blockBounds,blockBounds.size()*sizeof(blockBounds[0]));
//     memcpy(blockLevels.data(),_blockLevels,blockLevels.size()*sizeof(blockLevels[0]));
//     memcpy(blockOffsets.data(),_blockOffsets,blockOffsets.size()*sizeof(blockOffsets[0]));
//     memcpy(blockScalars.data(),_blockScalars,blockScalars.size()*sizeof(blockScalars[0]));
//     ScalarField::SP sf =
//       std::make_shared<BlockStructuredField>(checkGet(context),slot,
//                                              blockBounds,
//                                              blockLevels,
//                                              blockOffsets,
//                                              blockScalars);
//     return (BNScalarField)checkGet(context)->initReference(sf);
// #endif
//   }


  BARNEY_API
  BNGroup bnGroupCreate(BNContext _context,
                        int slot,
                        BNGeom *geoms, int numGeoms,
                        BNVolume *volumes, int numVolumes)
  {
    LOG_API_ENTRY;
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    Context *context = checkGet(_context);
    std::shared_ptr<Group>
      group = context->createGroup(slot,
                                   (Geometry **)geoms,numGeoms,
                                   (Volume **)volumes,numVolumes);
    // auto devices = context->getDevices(slot);
    // std::vector<Geometry::SP> _geoms;
    // for (int i=0;i<numGeoms;i++)
    //   _geoms.push_back(checkGet(geoms[i])->as<Geometry>());
    // std::vector<Volume::SP> _volumes;
    // for (int i=0;i<numVolumes;i++)
    //   _volumes.push_back(checkGet(volumes[i])->as<Volume>());
    // Group::SP group
    //   = std::make_shared<Group>(context,devices,
    //                             _geoms,_volumes);
    return (BNGroup)context->initReference(group);
    // } catch (std::runtime_error error) {
    //   std::cerr << "@barney: Error in creating group: " << error.what()
    //             << "... going to return null group." << std::endl;
    //   return BNGroup{0};
    // }
    BARNEY_LEAVE(__PRETTY_FUNCTION__,0);
  }

  BARNEY_API
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
  
  BARNEY_API
  void  bnBuild(BNModel model,
                int slot)
  {
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    LOG_API_ENTRY;
    checkGet(model)->build(slot);
    // checkGet(model,slot)->build();
    BARNEY_LEAVE(__PRETTY_FUNCTION__,);
  }
  
  BARNEY_API
  void bnCommit(BNObject target)
  {
    LOG_API_ENTRY;
    checkGet(target)->commit();
  }
  
              
  BARNEY_API
  void bnSetString(BNObject target, const char *param, const char *value)
  {
    if (!checkGet(target)->setString(checkGet(param),value))
      checkGet(target)->warn_unsupported_member(param,"std::string");
  }

  BARNEY_API
  void bnSetData(BNObject target, const char *param, BNData value)
  {
    if (!checkGet(target)->setData(checkGet(param),checkGetSP(value)))
      checkGet(target)->warn_unsupported_member(param,"BNData");
  }

  BARNEY_API
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

  BARNEY_API
  void bnSet1i(BNObject target, const char *param, int x)
  {
    if (!checkGet(target)->set1i(checkGet(param),x))
      checkGet(target)->warn_unsupported_member(param,"int");
  }

  BARNEY_API
  void bnSet2i(BNObject target, const char *param, int x, int y)
  {
    if (!checkGet(target)->set2i(checkGet(param),vec2i(x,y)))
      checkGet(target)->warn_unsupported_member(param,"vec2i");
  }

  BARNEY_API
  void bnSet3i(BNObject target, const char *param, int x, int y, int z)
  {
    if (!checkGet(target)->set3i(checkGet(param),vec3i(x,y,z)))
      checkGet(target)->warn_unsupported_member(param,"vec3i");
  }

# ifdef __VECTOR_TYPES__
  BARNEY_API
  void bnSet3ic(BNObject target, const char *param, int3 value)
  {
    if (!checkGet(target)->set3i(checkGet(param),(const vec3i&)value))
      checkGet(target)->warn_unsupported_member(param,"vec3i");
  }
#endif
  
  BARNEY_API
  void bnSet4i(BNObject target, const char *param, int x, int y, int z, int w)
  {
    if (!checkGet(target)->set4i(checkGet(param),vec4i(x,y,z,w)))
      checkGet(target)->warn_unsupported_member(param,"vec4i");
  }

  BARNEY_API
  void bnSet1f(BNObject target, const char *param, float value)
  {
    if (!checkGet(target)->set1f(checkGet(param),value))
      checkGet(target)->warn_unsupported_member(param,"float");
  }

  BARNEY_API
  void bnSet3f(BNObject target, const char *param, float x, float y, float z)
  {
    LOG_API_ENTRY;
    if (!checkGet(target)->set3f(checkGet(param),vec3f(x,y,z)))
      checkGet(target)->warn_unsupported_member(param,"vec3f");
  }

  BARNEY_API
  void bnSet4f(BNObject target, const char *param, float x, float y, float z, float w)
  {
    if (!checkGet(target)->set4f(checkGet(param),vec4f(x,y,z,w)))
      checkGet(target)->warn_unsupported_member(param,"vec4f");
  }

# ifdef __VECTOR_TYPES__
  BARNEY_API
  void bnSet3fc(BNObject target, const char *param, float3 value)
  {
    if (!checkGet(target)->set3f(checkGet(param),(const vec3f&)value))
      checkGet(target)->warn_unsupported_member(param,"vec3f");
  }

  BARNEY_API
  void bnSet4fc(BNObject target, const char *param, float4 value)
  {
    if (!checkGet(target)->set4f(checkGet(param),(const vec4f&)value))
      checkGet(target)->warn_unsupported_member(param,"vec4f");
  }
#endif
  
  BARNEY_API
  void bnSet4x3fv(BNObject target, const char *param, const BNTransform *transform)
  {
    assert(transform);
    if (!checkGet(target)->set4x3f(checkGet(param),*(const affine3f*)transform))
      checkGet(target)->warn_unsupported_member(param,"affine3f");
  }

  BARNEY_API
  void bnSet4x4fv(BNObject target, const char *param, const bn_float4 *transform)
  {
    assert(transform);
    if (!checkGet(target)->set4x4f(checkGet(param),(const vec4f*)transform))
      checkGet(target)->warn_unsupported_member(param,"mat4f");
  }
  



  

  
  BARNEY_API
  BNFrameBuffer bnFrameBufferCreate(BNContext _context,
                                    int owningRank)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<FrameBuffer> fb
      = context->createFrameBuffer(owningRank);
    return (BNFrameBuffer)context->initReference(fb);
  }

  BARNEY_API
  void bnFrameBufferResize(BNFrameBuffer fb,
                           int sizeX, int sizeY,
                           uint32_t channels)
  {
    LOG_API_ENTRY;
    checkGet(fb)->resize(vec2i{sizeX,sizeY},channels);
  }

  BARNEY_API
  void bnFrameBufferRead(BNFrameBuffer fb,
                         BNFrameBufferChannel channel,
                         void *hostPtr,
                         BNDataType requestedFormat)
  {
    LOG_API_ENTRY;
    checkGet(fb)->read(channel,hostPtr,requestedFormat);
  }
  
  BARNEY_API
  void *bnFrameBufferGetPointer(BNFrameBuffer fb,
                                BNFrameBufferChannel channel)
  {
    LOG_API_ENTRY;
    return checkGet(fb)->getPointer(channel);
  }
  

  BARNEY_API
  void bnAccumReset(BNFrameBuffer fb)
  {
    checkGet(fb)->resetAccumulation();
  }
  
  BARNEY_API
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

  BARNEY_API
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
    if (getenv("BARNEY_FORCE_CPU")) {
	    static int negOne = -1;
	    _gpuIDs = &negOne;
	    numGPUs = 1;
    }

    try {
      // ------------------------------------------------------------------
      // create vector of data groups; if actual specified by user we
      // use those; otherwise we use IDs
      // [0,1,...numModelSlotsOnThisHost)
      // ------------------------------------------------------------------
      assert(numDataRanksOnThisContext > 0);
      std::vector<int> dataGroupIDs;
      for (int i = 0;i < numDataRanksOnThisContext;i++)
        dataGroupIDs.push_back
          (dataRanksOnThisContext
           ? dataRanksOnThisContext[i]
           : i);

      // ------------------------------------------------------------------
      // create a backend. logic is as follows:
      //
      // 1) user can _explicitly_ request a CPU device by asking for a
      // single GPU with ID=-1 (ie, numGPUs=1,gpuIDs={-1}). If so,
      // create a CPU device if possible.
      // ------------------------------------------------------------------
      if (numGPUs == 1 && _gpuIDs[0] == -1) {
# if BARNEY_BACKEND_EMBREE
        return (BNContext)createContext_embree(dataGroupIDs);
# else
        throw std::runtime_error
          ("explicitly asked for CPU backend, "
           "but cpu/embree backend not compiled in");
# endif
      }

      // ------------------------------------------------------------------
      // 2) if user did specify a list of GPUs, create a GPU backend,
      // or return an error.
      // ------------------------------------------------------------------
      if (_gpuIDs != nullptr) {
#if BARNEY_BACKEND_OPTIX
        return (BNContext)createContext_optix(dataGroupIDs,numGPUs,_gpuIDs);
#else
        throw std::runtime_error
          ("explicitly asked for GPU backend, "
           "but optix support not compiled in");
#endif
      }

      // ------------------------------------------------------------------
      // 3) if user did not specify an explicit GPU list, try to
      // create a GPU backend, and fall back to embree if that doesn't
      // work.
      // ------------------------------------------------------------------

#if BARNEY_BACKEND_OPTIX
      try {
        PING; PRINT(numGPUs); 
        return (BNContext)createContext_optix(dataGroupIDs,numGPUs,_gpuIDs);
      } catch (std::exception &e) {
        std::cerr << "#barney(warn): could not create optix backend (reason: "
                  << e.what() << ")" << std::endl;
      }
#endif
      
# if BARNEY_BACKEND_EMBREE
      return (BNContext)createContext_embree(dataGroupIDs);
#endif
      throw std::runtime_error("could not generate _any_ backend?!");
      
#if 0
      // ------------------------------------------------------------------
      // create list of GPUs to use for this rank. if specified by user
      // we use this; otherwise we use GPUs in order, split into groups
      // according to how many ranks there are on this host. Ie, if host
      // has four GPUs the first rank will take 0 and 1; and the second
      // one will take 2 and 3.
      // ------------------------------------------------------------------
      std::vector<int> gpuIDs;
      if (_gpuIDs) {
        for (int i = 0;i < numGPUs;i++)
          gpuIDs.push_back(_gpuIDs[i]);
      }
      else {
        if (numGPUs < 1)
          numGPUs = rtc::Backend::getDeviceCount();
        // cudaGetDeviceCount(&numGPUs);
        for (int i = 0;i < numGPUs;i++)
          gpuIDs.push_back(i);
      }
      if (gpuIDs.empty())
        throw std::runtime_error
          ("no devices found!?");

      if (gpuIDs.size() < numDataRanksOnThisContext) {
        std::vector<int> replicatedIDs;
        for (int i = 0;i < numDataRanksOnThisContext;i++)
          replicatedIDs.push_back(gpuIDs[i % gpuIDs.size()]);
        gpuIDs = replicatedIDs;
      }

      return (BNContext)new LocalContext(dataGroupIDs,
                                         gpuIDs);
#endif
    } 
    catch (const std::exception& e) {
      std::cerr << "error creating barney context : " << e.what() << std::endl;
      return 0;
    }
    return 0;
  }

#if BARNEY_MPI
  BARNEY_API
  BNContext bnMPIContextCreate(MPI_Comm _comm,
                               /*! how many data slots this context is to
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
                               int  numGPUs
                               )
  {
    LOG_API_ENTRY;

    mpi::Comm world(_comm);

    if (world.size == 1) {
      // std::cout << "#bn: MPIContextInit, but only one rank - using local context" << std::endl;
      return bnContextCreate(dataRanksOnThisContext,
                             numDataRanksOnThisContext == 0
                             ? 1 : numDataRanksOnThisContext,
                             /*! which gpu(s) to use for this
                               process. default is to distribute
                               node's GPUs equally over all ranks on
                               that given node */
                             _gpuIDs,
                             numGPUs);
    }

    // ------------------------------------------------------------------
    // create vector of data groups; if actual specified by user we
    // use those; otherwise we use IDs
    // [0,1,...numModelSlotsOnThisHost)
    // ------------------------------------------------------------------
    assert(/* data groups == 0 is allowed for passive nodes*/
           numDataRanksOnThisContext >= 0);
    std::vector<int> dataGroupIDs;
    int rank;
    MPI_Comm_rank(world, &rank);
    for (int i=0;i<numDataRanksOnThisContext;i++)
      dataGroupIDs.push_back
        (dataRanksOnThisContext
         ? dataRanksOnThisContext[i]
         : rank*numDataRanksOnThisContext+i);

    // check if we're an active worker
    bool isActiveWorker = !dataGroupIDs.empty();
    mpi::Comm workers = world.split(isActiveWorker);
    
    // ------------------------------------------------------------------
    // create list of GPUs to use for this rank. if specified by user
    // we use this; otherwise we use GPUs in order, split into groups
    // according to how many ranks there are on this host. Ie, if host
    // has four GPUs the first rank will take 0 and 1; and the second
    // one will take 2 and 3.
    // ------------------------------------------------------------------
    BNHardwareInfo hardware;
    bnMPIQueryHardware(&hardware,_comm);

    if (_gpuIDs) {
      // gpu IDs _are_ specified by user - use them, or fail
      assert(numGPUs > 0);
      if (numGPUs == 1 && _gpuIDs[0] == -1) {
# if BARNEY_BACKEND_EMBREE
        return (BNContext)createMPIContext_embree(world,
                                                  workers,
                                                  isActiveWorker,
                                                  dataGroupIDs);
# else
        throw std::runtime_error
          ("explicitly asked for CPU backend, "
           "but cpu/embree backend not compiled in");
# endif
      }
#if BARNEY_BACKEND_OPTIX
      std::vector<int> gpuIDs = {_gpuIDs,_gpuIDs+numGPUs};
      return (BNContext)createMPIContext_optix(world,
                                               workers,
                                               isActiveWorker,
                                               dataGroupIDs,
                                               gpuIDs);
#else
      throw std::runtime_error("explicitly asked for gpus to use, "
                               "but optix backend not compiled in");
#endif
    }

    
    // user did NOT request any GPUs, it's up to us. 
#if BARNEY_BACKEND_OPTIX
    try {
      if (hardware.numGPUsThisRank == 0)
        throw std::runtime_error("don't have any GPUs on this node");

      std::vector<int> gpuIDs;      
      for (int i=0;i<hardware.numGPUsThisRank;i++)
        gpuIDs.push_back((hardware.localRank*hardware.numGPUsThisRank
                          + i) % hardware.numGPUsThisHost);
      return (BNContext)createMPIContext_optix(world,
                                               workers,
                                               isActiveWorker,
                                               dataGroupIDs,
                                               gpuIDs);
    } catch (std::exception &e) {
      std::cout << "#barney: could not create optix context" << std::endl;
    }
#endif
    
# if BARNEY_BACKEND_EMBREE
    return (BNContext)createMPIContext_embree(world,
                                              workers,
                                              isActiveWorker,
                                              dataGroupIDs);
# else
    throw std::runtime_error
      ("cannot find GPUs for otpix backend, but cpu backend not compiled in");
# endif
  }
#endif
} // ::owl
