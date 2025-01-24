#include "rtcore/embree/Device.h"
#include "rtcore/embree/Compute.h"
#include "rtcore/embree/Texture.h"
#include "rtcore/embree/Buffer.h"
#include "rtcore/common/RTCore.h"

namespace barney {
  namespace embree {

    // ------------------------------------------------------------------
    // rt core interface
    // ------------------------------------------------------------------
    void TraceInterface::ignoreIntersection() const
      { BARNEY_NYI(); }
      void TraceInterface::reportIntersection(float t, int i) const
      { BARNEY_NYI(); }
      void *TraceInterface::getPRD() const
      { BARNEY_NYI(); }
      const void *TraceInterface::getProgramData() const
      { BARNEY_NYI(); }
      const void *TraceInterface::getLPData() const
      { BARNEY_NYI(); }
      vec3i TraceInterface::getLaunchDims()  const
      { BARNEY_NYI(); }
      vec3i TraceInterface::getLaunchIndex() const
      { BARNEY_NYI(); }
      vec2f TraceInterface::getTriangleBarycentrics() const
      { BARNEY_NYI(); }
      int TraceInterface::getPrimitiveIndex() const
      { BARNEY_NYI(); }
      float TraceInterface::getRayTmax() const
      { BARNEY_NYI(); }
      float TraceInterface::getRayTmin() const
      { BARNEY_NYI(); }
      vec3f TraceInterface::getObjectRayDirection() const
      { BARNEY_NYI(); }
      vec3f TraceInterface::getObjectRayOrigin() const
      { BARNEY_NYI(); }
      vec3f TraceInterface::getWorldRayDirection() const
      { BARNEY_NYI(); }
      vec3f TraceInterface::getWorldRayOrigin() const
      { BARNEY_NYI(); }
      vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
      { BARNEY_NYI(); }
      vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
      { BARNEY_NYI(); }
      vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
      { BARNEY_NYI(); }
      vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
      { BARNEY_NYI(); }
      vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
      { BARNEY_NYI(); }
      vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
      { BARNEY_NYI(); }
      void  TraceInterface::traceRay(rtc::device::AccelHandle world,
                     vec3f org,
                     vec3f dir,
                     float t0,
                     float t1,
                     void *prdPtr) const
      { BARNEY_NYI(); }
      
    
    


    // ------------------------------------------------------------------
    // device
    // ------------------------------------------------------------------
    
    Device::Device(int physicalGPU)
      : rtc::Device(physicalGPU)
    {
      embreeDevice = rtcNewDevice("verbose=0");
    }

    Device::~Device()
    {
      destroy();
    }

    void Device::destroy()
    {
      rtcReleaseDevice(embreeDevice);
      embreeDevice = 0;
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
    { return new Trace(this,name,rayGenSize); }
    
    
  }
}

