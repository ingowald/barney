#pragma once

#include "rtcore/common/Backend.h"
#include "embree4/rtcore.h"

namespace barney {
  namespace embree {

    struct Device : public rtc::Device {
      Device(int physicalGPU);
      virtual ~Device();

      void destroy() override;

      // ==================================================================
      // basic compute stuff
      // ==================================================================
      void copyAsync(void *dst, const void *src, size_t numBytes) override
      { memcpy(dst,src,numBytes); }
      
      void *allocHost(size_t numBytes) override
      { return malloc(numBytes); }
      
      void freeHost(void *mem) override
      { free(mem); }
      
      void memsetAsync(void *mem,int value, size_t size) override
      { memset(mem,value,size); }
      
      void *alloc(size_t numBytes) override
      { return malloc(numBytes); }
      
      void freeMem(void *mem) override
      { free(mem); }
      
      void sync() override
      {/*no-op*/}
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const override
      {/*no-op*/ return 0; }
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const override
      {/*no-op*/}

      rtc::TextureData *createTextureData(vec3i dims,
                                          rtc::DataType format,
                                          const void *texels) override;
      void freeTextureData(rtc::TextureData *td) override;
      void freeTexture(rtc::Texture *tex) override;
      
      // ==================================================================
      // kernels
      // ==================================================================
      rtc::Compute *createCompute(const std::string &name) override;
      
      rtc::Trace *createTrace(const std::string &name,
                              size_t rayGenSize) override;

      // ==================================================================
      // buffer stuff
      // ==================================================================
      rtc::Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0) override;
      
      void freeBuffer(rtc::Buffer *buffer) override;
      
      // ==================================================================
      // texture stuff
      // ==================================================================

      // in parent

      // ==================================================================
      // ray tracing pipeline related stuff
      // ==================================================================


      // ------------------------------------------------------------------
      // rt pipeline/sbtstuff
      // ------------------------------------------------------------------

      void buildPipeline() override
      { BARNEY_NYI(); }
      
      void buildSBT() override
      { BARNEY_NYI(); }

      // ------------------------------------------------------------------
      // geomtype stuff
      // ------------------------------------------------------------------
      
      rtc::GeomType *
      createUserGeomType(const char *typeName,
                         size_t sizeOfDD,
                         const char *boundsFctName,
                         const char *isecFctName,
                         const char *ahFctName,
                         const char *chFctName) 
        override 
      { BARNEY_NYI(); }
      
      rtc::GeomType *
      createTrianglesGeomType(const char *typeName,
                              size_t sizeOfDD,
                              bool has_ah,
                              bool has_ch) override
      { BARNEY_NYI(); }
      
      void freeGeomType(rtc::GeomType *) override 
      { BARNEY_NYI(); }

      // ------------------------------------------------------------------
      // geom stuff
      // ------------------------------------------------------------------
      
      void freeGeom(rtc::Geom *) override 
      { BARNEY_NYI(); }
      
      // ------------------------------------------------------------------
      // group/accel stuff
      // ------------------------------------------------------------------
      rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms) override
      { BARNEY_NYI(); }
      
      rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) override
      { BARNEY_NYI(); }

      rtc::Group *
      createInstanceGroup(const std::vector<rtc::Group *> &groups,
                          const std::vector<affine3f> &xfms) override
      { BARNEY_NYI(); }

      void freeGroup(rtc::Group *group) override
      { delete group; }

      RTCDevice embreeDevice = 0;
    };
    
  }
}
