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

#include "rtcore/embree/Device.h"
#include "rtcore/embree/Compute.h"
#include "rtcore/embree/Texture.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/embree/Triangles.h"
#include "rtcore/embree/UserGeom.h"
#include "rtcore/embree/Group.h"
// ---- common ----
#include "rtcore/common/RTCore.h"

namespace barney {
  namespace embree {

    __thread TraceInterface *perThreadTraceInterface = 0;
    
    // TraceInterface *TraceInterface::get()
    // {
    //   // if (!perThreadTraceInterface)
    //   //   perThreadTraceInterface = new TraceInterface;
    //   return perThreadTraceInterface;
    // }

    // ------------------------------------------------------------------
    // rt core interface
    // ------------------------------------------------------------------
    void TraceInterface::ignoreIntersection() 
    { ignoreThisHit = true;  }
    void TraceInterface::reportIntersection(float t, int i)
    { isec_t = t;  }
    void *TraceInterface::getPRD() const
    { return prd; }
    const void *TraceInterface::getProgramData() const
    { return geomData; }
    const void *TraceInterface::getLPData() const
    { return lpData; }
    vec3i TraceInterface::getLaunchDims()  const
    { return launchDimensions; }
    vec3i TraceInterface::getLaunchIndex() const
    { return launchIndex; }
    vec2f TraceInterface::getTriangleBarycentrics() const
    { return triangleBarycentrics; }
    int TraceInterface::getPrimitiveIndex() const
    { return primID; }
    float TraceInterface::getRayTmax() const
    { return embreeRay->tfar; }
    float TraceInterface::getRayTmin() const
    { return embreeRay->tnear; }
    vec3f TraceInterface::getObjectRayDirection() const
    { return *(vec3f*)&embreeRay->dir_x; }
    vec3f TraceInterface::getObjectRayOrigin() const
    { return *(vec3f*)&embreeRay->org_x; }
    vec3f TraceInterface::getWorldRayDirection() const
    { return worldDirection; }
    vec3f TraceInterface::getWorldRayOrigin() const
    { return worldOrigin; }
    vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
    {
      return xfmVector(objectToWorldXfm->l,
                         (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmPoint(*objectToWorldXfm,
                        (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmVector(objectToWorldXfm->l,
                         (const owl::common::vec3f &)v);
    }

    vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
    {
      return xfmVector(worldToObjectXfm->l,
                         (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmPoint(*worldToObjectXfm,
                        (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmVector(worldToObjectXfm->l,
                         (const owl::common::vec3f &)v);
    }

    void intersectionFilter(const RTCFilterFunctionNArguments* args)
    {
      /* avoid crashing when debug visualizations are used */
      if (args->context == nullptr) return;

      assert(args->N == 1);
      int* valid = args->valid;
      if (valid[0] != -1) return;
      // const RTCRayQueryContext* context = (const RTCRayQueryContext*) args->context;
      RTCRay* ray = (RTCRay*)args->ray;
      RTCHit* hit = (RTCHit*)args->hit;
  
      // TraceInterface *ti2 = (TraceInterface *)TraceInterface::get();//args->context;
      TraceInterface *ti = (TraceInterface *)args->context;

      int primID = hit->primID;
      int geomID = hit->geomID;
      int instID = hit->instID[0];

      InstanceGroup *ig = ti->world;
      GeomGroup *group = ig->getGroup(instID);
      Geom *geom = (Geom*)group->getGeom(geomID);
      GeomType *gt = geom->type;
      if (gt->ah) {
        ti->geomData = (void*)geom->programData.data();
        ti->ignoreThisHit = false;
        ti->primID = primID;
        ti->geomID = geomID;
        ti->instID = instID;
        ti->triangleBarycentrics = { hit->u,hit->v };
        ti->objectToWorldXfm = &ig->xfms[instID];
        ti->worldToObjectXfm = &ig->inverseXfms[instID];
        ti->embreeRay = ray;
        ti->embreeHit = hit;
    
        gt->ah(*ti);

        if (ti->ignoreThisHit) {
          valid[0] = 0;
          return;
        } 
      }
    }

    
    __both__ void TraceInterface::traceRay(rtc::device::AccelHandle world,
                                   vec3f rayOrigin,
                                   vec3f rayDirection,
                                   float tmin,
                                   float tmax,
                                   void *prdPtr) 
    {
      perThreadTraceInterface = this;
      InstanceGroup *ig = (InstanceGroup *)world;
      RTCScene embreeScene = ig->embreeScene;
      assert(embreeScene);

      RTCRayHit rayHit;
  
      TraceInterface *ti = this;//TraceInterface::get();
      // LaunchContext *lc = LaunchContext::get();
      ti->world = ig;

      ti->worldOrigin = rayOrigin;
      ti->worldDirection = rayDirection;
      ti->prd = prdPtr;

      rayHit.ray.org_x = rayOrigin.x;        // x coordinate of ray origin
      rayHit.ray.org_y = rayOrigin.y;        // y coordinate of ray origin
      rayHit.ray.org_z = rayOrigin.z;        // z coordinate of ray origin
      rayHit.ray.tnear = tmin;        // start of ray segment
  
      rayHit.ray.dir_x = rayDirection.x;        // x coordinate of ray direction
      rayHit.ray.dir_y = rayDirection.y;        // y coordinate of ray direction
      rayHit.ray.dir_z = rayDirection.z;        // z coordinate of ray direction
      rayHit.ray.time = 0;         // time of this ray for motion blur
  
      rayHit.ray.tfar = tmax;         // end of ray segment (set to hit distance)
      
      rayHit.ray.mask = -1;
      rayHit.ray.flags = 0;
      rayHit.hit.primID    = RTC_INVALID_GEOMETRY_ID;
      rayHit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
      rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
      
      ti->embreeRay = &rayHit.ray;
      ti->embreeHit = &rayHit.hit;
      
      rtcInitRayQueryContext(&ti->embreeRayQueryContext);
      ti->world = ig;
  
      /* intersect ray with scene */
      RTCIntersectArguments iargs;
      rtcInitIntersectArguments(&iargs);
      iargs.context = &ti->embreeRayQueryContext;
      iargs.filter = intersectionFilter;

      rtcIntersect1(embreeScene,&rayHit,&iargs);

      if ((int)rayHit.hit.geomID >= 0) {
    
        int primID = rayHit.hit.primID;
        int geomID = rayHit.hit.geomID;
        int instID = rayHit.hit.instID[0];
    
        GeomGroup *group = (GeomGroup *)ig->groups[instID];
        Geom *geom = (Geom *)group->geoms[geomID];
        GeomType *gt = geom->type;
        if (gt->ch) {

          ti->geomData = (void*)geom->programData.data();
          ti->primID = primID;
          ti->geomID = geomID;
          ti->instID = instID;
          ti->triangleBarycentrics = { rayHit.hit.u,rayHit.hit.v };
          ti->objectToWorldXfm = &ig->xfms[instID];
          ti->worldToObjectXfm = &ig->inverseXfms[instID];
          ti->embreeRay = &rayHit.ray;
          ti->embreeHit = &rayHit.hit;
    
          gt->ch(*ti);
        }
      }
      
    }
      
    

    // ------------------------------------------------------------------
    // device
    // ------------------------------------------------------------------
    
    Device::Device(int physicalGPU)
      : rtc::Device(physicalGPU)
    {
      embreeDevice = rtcNewDevice("verbose=0");
      ls = createLaunchSystem();
    }

    Device::~Device()
    {
      // todo: shut down launch system
      destroy();
    }

    void Device::destroy()
    {
      rtcReleaseDevice(embreeDevice);
      embreeDevice = 0;
    }


    // ------------------------------------------------------------------
    // group/accel stuff
    // ------------------------------------------------------------------

    rtc::Group *
    Device::createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
    { return new TrianglesGroup(this,geoms); }
    
    rtc::Group *
    Device::createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) 
    { return new UserGeomGroup(this,geoms); }
      
    rtc::Group *
    Device::createInstanceGroup(const std::vector<rtc::Group *> &groups,
                                const std::vector<affine3f> &xfms) 
    { return new InstanceGroup(this,groups,xfms); }
    
    void Device::freeGroup(rtc::Group *group) 
    { delete group; }

    // ------------------------------------------------------------------
    // geom stuff
    // ------------------------------------------------------------------


    rtc::GeomType *Device::createTrianglesGeomType(const char *typeName,
                                                   size_t sizeOfDD,
                                                   bool has_ah,
                                                   bool has_ch)
    {
      return new TrianglesGeomType(this,typeName,sizeOfDD,has_ah,has_ch);
    }
    
    void Device::freeGeomType(rtc::GeomType *gt) 
    {
      delete gt;
    }
    
    rtc::TextureData *Device::createTextureData(vec3i dims,
                                                rtc::DataType format,
                                                const void *texels) 
    { return new TextureData(this,dims,format,texels); }

    rtc::Buffer *Device::createBuffer(size_t numBytes,
                                      const void *initValues) 
    {
      return new Buffer(this,numBytes,initValues);
    }

    void Device::freeTextureData(rtc::TextureData *td) 
    { delete td; }
      
    void Device::freeTexture(rtc::Texture *tex) 
    { delete tex; }


    void Device::freeBuffer(rtc::Buffer *buffer) 
    {
      delete buffer;
    }
    
    rtc::Compute *Device::createCompute(const std::string &name) 
    { return new Compute(this,name); }
    
    
    rtc::Trace *Device::createTrace(const std::string &name,
                                    size_t rayGenSize) 
    { return new Trace(this,name); }
    
    void Device::buildPipeline()
    {}
    
    void Device::buildSBT()
    {}
      

    
  }
}

