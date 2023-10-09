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

namespace barney {

  inline Context *checkGet(BNContext context)
  {
    assert(context);
    return (Context *)context;
  }
  
  inline Model *checkGet(BNModel model)
  {
    assert(model);
    return (Model *)model;
  }
  
  inline FrameBuffer *checkGet(BNFrameBuffer frameBuffer)
  {
    assert(frameBuffer);
    return (FrameBuffer *)frameBuffer;
  }
  
  BN_API
  BNModel bnModelCreate(BNContext ctx)
  {
    return (BNModel)checkGet(ctx)->createModel();
  }

  BN_API
  BNDataGroup bnGetDataGroup(BNModel model,
                             int dataGroupID)
  {
    WARN_NOTIMPLEMENTED;
    return 0;
  }
  
  BN_API
  void bnModelSetInstances(BNDataGroup dataGroup,
                           BNGroup *groups,
                           int numGroups,
                           BNVolume *volumes,
                           int numVolumes)
  {
    WARN_NOTIMPLEMENTED;
  }
  

  BN_API
  void bnContextDestroy(BNContext context)
  {
    delete (Context *)context;
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
    return 0;
  }

  BN_API
  BNGroup bnGroupCreate(BNDataGroup dataGroup,
                        BNGeom *geoms, int numGeoms,
                        BNVolume *volumes, int numVolumes)
  {
    return 0;
  }
  
  BN_API
  void  bnModelBuild(BNModel model)
  {
  }

  BN_API
  void bnPinholeCamera(BNCamera *camera,
                       float3 _from,
                       float3 _at,
                       float3 _up,
                       float  fov,
                       int2   fbSize)
  {
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
  }
  
  BN_API
  BNFrameBuffer bnFrameBufferCreate(BNContext context,
                                    int owningRank)
  {
    FrameBuffer *fb = checkGet(context)->createFB(owningRank);
    // fb->resize({sizeX,sizeY});
    return (BNFrameBuffer)fb;
  }

  BN_API
  void bnFrameBufferResize(BNFrameBuffer fb,
                           int sizeX, int sizeY,
                           uint32_t *hostRGBA)
  {
    checkGet(fb)->resize(vec2i{sizeX,sizeY},hostRGBA);
  }

  BN_API
  void bnRender(BNModel model,
                const BNCamera *camera,
                BNFrameBuffer fb,
                BNRenderRequest *req)
  {
    assert(camera);
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
    
    return (BNContext)new LocalContext(dataGroupIDs,
                                       gpuIDs);
  }
}
