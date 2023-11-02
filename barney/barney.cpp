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

#include "barney.h"
#include "barney/Context.h"
#include "barney/LocalContext.h"
#include "barney/FrameBuffer.h"
#include "barney/Model.h"

#define WARN_NOTIMPLEMENTED std::cout << " ## " << __PRETTY_FUNCTION__ << " not implemented yet ..." << std::endl;

#if 1
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
  
  inline Material checkGet(const BNMaterial *material)
  {
    assert(material);
    Material result;
    result.diffuseColor = (const vec3f&)material->baseColor;
    return result;
  }
  
  inline DataGroup *checkGet(BNDataGroup dg)
  {
    assert(dg);
    return (DataGroup *)dg;
  }
  
  inline Model *checkGet(BNModel model)
  {
    assert(model);
    return (Model *)model;
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
  // inline TransferFunction::SP checkGetSP(BNTransferFunction xf)
  // {
  //   return checkGet(xf)->shared_from_this()->as<TransferFunction>();
  // }

  // ------------------------------------------------------------------
  
  BN_API
  BNModel bnModelCreate(BNContext ctx)
  {
    LOG_API_ENTRY;
    return (BNModel)checkGet(ctx)->createModel();
  }

  BN_API
  BNDataGroup bnGetDataGroup(BNModel model,
                             int dataGroupID)
  {
    LOG_API_ENTRY;
    return (BNDataGroup)checkGet(model)->getDG(dataGroupID);
  }
  
  BN_API
  void bnModelSetInstances(BNDataGroup dataGroup,
                           BNGroup *_groups,
                           BNTransform *xfms,
                           int numInstances)
  {
    LOG_API_ENTRY;
    
    std::vector<Group::SP> groups;
    for (int i=0;i<numInstances;i++) {
      groups.push_back(checkGetSP(_groups[i]));
    }
    checkGet(dataGroup)->setInstances(groups,(const affine3f *)xfms);
  }
  

  BN_API
  void bnContextDestroy(BNContext context)
  {
    LOG_API_ENTRY;
    delete (Context *)context;
  }

  BN_API
  BNGeom bnSpheresCreate(BNDataGroup       dataGroup,
                         const BNMaterial *material,
                         const float3     *origins,
                         int               nubarneygins,
                         const float      *radii,
                         float             defaultRadius)
  {
    LOG_API_ENTRY;
    Spheres *spheres = checkGet(dataGroup)->createSpheres
      (checkGet(material),(const vec3f*)origins,nubarneygins,radii,defaultRadius);
    return (BNGeom)spheres;
  }

  BN_API
  BNGeom bnTriangleMeshCreate(BNDataGroup dataGroup,
                              const BNMaterial *material,
                              const int3 *indices,
                              int numIndices,
                              const float3 *vertices,
                              int numVertices,
                              const float3 *normals,
                              const float2 *texcoords)
  {
    // LOG_API_ENTRY;
    Triangles *triangles = checkGet(dataGroup)->createTriangles
      (checkGet(material),
       numIndices,
       (const vec3i*)indices,
       numVertices,
       (const vec3f*)vertices,
       (const vec3f*)normals,
       (const vec2f*)texcoords);
    return (BNGeom)triangles;
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
  BNVolume bnVolumeCreate(BNDataGroup dataGroup,
                          BNScalarField _sf)
  {
    ScalarField::SP sf = checkGetSP(_sf);
    return (BNVolume)checkGet(dataGroup)->createVolume(sf);
  }

  BN_API
  BNScalarField bnUMeshCreate(BNDataGroup dataGroup,
                              // vertices, 4 floats each (3 floats position,
                              // 4th float scalar value)
                              const float *_vertices, int numVertices,
                              // tets, 4 ints in vtk-style each
                              const int *_tetIndices, int numTets,
                              // pyramids, 5 ints in vtk-style each
                              const int *_pyrIndices, int numPyrs,
                              // wedges/tents, 6 ints in vtk-style each
                              const int *_wedIndices, int numWeds,
                              // general (non-guaranteed cube/voxel) hexes, 8
                              // ints in vtk-style each
                              const int *_hexIndices, int numHexes)
  {
    std::cout << "#bn: copying umesh from app ..." << std::endl;
    std::vector<vec4f>      vertices(numVertices);
    std::vector<TetIndices> tetIndices(numTets);
    std::vector<PyrIndices> pyrIndices(numPyrs);
    std::vector<WedIndices> wedIndices(numWeds);
    std::vector<HexIndices> hexIndices(numHexes);
    memcpy(tetIndices.data(),_tetIndices,tetIndices.size()*sizeof(tetIndices[0]));
    memcpy(pyrIndices.data(),_pyrIndices,pyrIndices.size()*sizeof(pyrIndices[0]));
    memcpy(wedIndices.data(),_wedIndices,wedIndices.size()*sizeof(wedIndices[0]));
    memcpy(hexIndices.data(),_hexIndices,hexIndices.size()*sizeof(hexIndices[0]));
    memcpy(vertices.data(),_vertices,vertices.size()*sizeof(vertices[0]));
    ScalarField *sf = checkGet(dataGroup)->createUMesh(vertices,
                                                       tetIndices,
                                                       pyrIndices,
                                                       wedIndices,
                                                       hexIndices);
    return (BNScalarField)sf;
  }
  

  BN_API
  BNGroup bnGroupCreate(BNDataGroup dataGroup,
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
    Group *group = checkGet(dataGroup)->createGroup(_geoms,_volumes);
    return (BNGroup)group;
  }

  BN_API
  void  bnGroupBuild(BNGroup group)
  {
    LOG_API_ENTRY;
    checkGet(group)->build();
  }
  
  BN_API
  void  bnModelBuild(BNDataGroup dataGroup)
  {
    LOG_API_ENTRY;
    checkGet(dataGroup)->build();
  }
  // BN_API
  // void  bnModelBuild(BNModel model)
  // {
  //   checkGet(model)->build();
  // }

  BN_API
  void bnPinholeCamera(BNCamera *camera,
                       float3 _from,
                       float3 _at,
                       float3 _up,
                       float  fov,
                       int2   fbSize)
  {
    // LOG_API_ENTRY;
    assert(camera);
    vec3f from = (const vec3f&)_from;
    vec3f at   = (const vec3f&)_at;
    vec3f up   = (const vec3f&)_up;
    
    vec3f dir_00  = normalize(at-from);
    
    vec3f dir_du = normalize(cross(dir_00, up));
    vec3f dir_dv = normalize(cross(dir_du, dir_00));

    float min_xy = (float)std::min(fbSize.x, fbSize.y);

    dir_00 *= (float)((float)min_xy / (2.0f * tanf((0.5f * fov) * M_PI / 180.0f)));
    dir_00 -= 0.5f * (float)fbSize.x * dir_du;
    dir_00 -= 0.5f * (float)fbSize.y * dir_dv;

    camera->dir_00 = (float3&)dir_00;
    camera->dir_du = (float3&)dir_du;
    camera->dir_dv = (float3&)dir_dv;
    camera->lens_00 = (float3&)from;
    camera->lensRadius = 0.f;
    camera->dbg_vi = _at;
    camera->dbg_vp = _from;
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
                           uint32_t *hostRGBA)
  {
    LOG_API_ENTRY;
    checkGet(fb)->resize(vec2i{sizeX,sizeY},hostRGBA);
  }


  BN_API
  void bnAccumReset(BNFrameBuffer fb)
  {
    checkGet(fb)->resetAccumulation();
  }
  
  BN_API
  void bnRender(BNModel model,
                const BNCamera *_camera,
                BNFrameBuffer fb,
                BNRenderRequest *req)
  {
    // std::cout << "------------------------------------------------------------------ " << std::endl;
    
    static int count = 0;
    if (count++ < 3)
      LOG_API_ENTRY;

    // if (count > 2) exit(0);

    assert(_camera);
    Camera camera;
    camera.dbg_vi = (const vec3f&)_camera->dbg_vi;
    camera.dbg_vp = (const vec3f&)_camera->dbg_vp;
    camera.lens_00 = (const vec3f&)_camera->lens_00;
    camera.dir_00 = (const vec3f&)_camera->dir_00;
    camera.dir_du = (const vec3f&)_camera->dir_du;
    camera.dir_dv = (const vec3f&)_camera->dir_dv;
    camera.lensRadius = _camera->lensRadius;
    checkGet(model)->render(camera,checkGet(fb));
  }

  BN_API
  BNContext bnContextCreate(const int *dataGroupsOnThisRank,
                            int  numDataGroupsOnThisRank,
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
    // [0,1,...numDataGroupsOnThisHost)
    // ------------------------------------------------------------------
    assert(numDataGroupsOnThisRank > 0);
    std::vector<int> dataGroupIDs;
    for (int i=0;i<numDataGroupsOnThisRank;i++)
      dataGroupIDs.push_back
        (dataGroupsOnThisRank
         ? dataGroupsOnThisRank[i]
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

    if (gpuIDs.size() < numDataGroupsOnThisRank) {
      std::vector<int> replicatedIDs;
      for (int i=0;i<numDataGroupsOnThisRank;i++)
        replicatedIDs.push_back(gpuIDs[i % gpuIDs.size()]);
      gpuIDs = replicatedIDs;
    }
    
    return (BNContext)new LocalContext(dataGroupIDs,
                                       gpuIDs);
  }
}
