#pragma once

#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace optix {
    struct OptixBackend;
    struct DevGroup;
    
    struct OptixDevice {
      OptixDevice(DevGroup *parent,
                  int physicalGPU,
                  size_t sizeOfGlobals);
      virtual ~OptixDevice();
      
      OWLContext      owl = 0;
      OWLLaunchParams lp  = 0;
      OWLRayGen       rg  = 0;

      DevGroup *const parent;
      int       const physicalGPU;
    };
    
    struct DevGroup : public cuda::BaseDevGroup {
      DevGroup(OptixBackend *backend,
               const std::vector<int> &gpuIDs,
               size_t sizeOfGlobals);
      virtual ~DevGroup();

      void destroy() 
        override
      { BARNEY_NYI(); };
      
      
      void free(rtc::GeomType *) 
        override 
      { BARNEY_NYI(); };

      void free(rtc::Geom *)
        override 
      { BARNEY_NYI(); };
      rtc::Geom *
      createGeom(rtc::GeomType *gt) 
        override 
      { BARNEY_NYI(); };
      
      rtc::GeomType *
      createUserGeomType(const char *typeName,
                         size_t sizeOfDD,
                         const char *boundsFctName,
                         const char *isecFctName,
                         const char *ahFctName,
                         const char *chFctName) 
        override 
      { BARNEY_NYI(); };
      
      rtc::GeomType *
      createTrianglesGeomType(const char *typeName,
                              size_t sizeOfDD,
                              const char *ahFctName,
                              const char *chFctName) 
        override 
      { BARNEY_NYI(); };
      

      // ==================================================================
      // group/accel stuff
      // ==================================================================
      rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
        override
      { BARNEY_NYI(); };
      
      rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) 
        override
      { BARNEY_NYI(); };

      rtc::Group *
      createInstanceGroup(const std::vector<rtc::Group *> &groups,
                          const std::vector<affine3f> &xfms) 
        override 
      { BARNEY_NYI(); };

      void free(rtc::Group *)
        override 
      { BARNEY_NYI(); };

      // ==================================================================
      // texture stuff
      // ==================================================================


      rtc::TextureData *
      createTextureData(vec3i dims,
                        rtc::TextureData::Format format,
                        const void *texels) const
        override 
      { BARNEY_NYI(); };
                        
      rtc::Texture *
      createTexture(rtc::TextureData *data,
                    rtc::Texture::FilterMode filterMode,
                    rtc::Texture::AddressMode addressModes[3],
                    const vec4f borderColor = vec4f(0.f),
                    rtc::Texture::ColorSpace colorSpace
                    = rtc::Texture::COLOR_SPACE_LINEAR) const
        override 
      { BARNEY_NYI(); };
                        
      void free(rtc::TextureData *) const
        override 
      { BARNEY_NYI(); };
      void free(rtc::Texture *) const 
        override 
      { BARNEY_NYI(); };


      // ==================================================================
      // buffer stuff
      // ==================================================================
      rtc::Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0) const 
        override 
      { BARNEY_NYI(); };
      void free(rtc::Buffer *) const 
        override 
      { BARNEY_NYI(); };
      void copy(rtc::Buffer *dst,
                rtc::Buffer *src,
                size_t numBytes) const 
        override 
      { BARNEY_NYI(); };
        
        
      
      std::vector<OptixDevice *> devices;
    };
    
    struct OptixBackend : public cuda::BaseBackend {
      // setActive and getActive: inherited from cuda bacekdn
      OptixBackend();
      virtual ~OptixBackend() = default;
      
      rtc::Device *createDevice(int physicalID) override;
      
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                    size_t sizeOfGlobals) override;
    };
    
  }
}

  
  
