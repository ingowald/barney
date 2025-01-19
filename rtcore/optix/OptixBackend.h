#pragma once

#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace optix {
    struct OptixBackend;
    struct DevGroup;
    struct Device;
    
    struct OptixDevice : public cuda::BaseDevice {
      OptixDevice(const optix::DevGroup *parent,
                  int physicalGPU,
                  int owlID);
      virtual ~OptixDevice();

      const DevGroup *const parent;
      // OWLLaunchParams lp  = 0;
      // OWLRayGen       rg  = 0;

      // DevGroup *const parent;
      // int       const physicalGPU;
    };

    struct Buffer : public rtc::Buffer {
      Buffer(const optix::DevGroup *dg,
                  size_t size,
                  const void *initData);
      void *getDD(const rtc::Device *) const override;
      void upload(const void *hostPtr,
                  size_t numBytes,
                  size_t ofs = 0,
                  const rtc::Device *device=nullptr) override;
      OWLBuffer owl;
    };

    struct Geom : public rtc::Geom {
      Geom(OWLGeom geom);
      virtual ~Geom();
      void setDD(const void *dd, const rtc::Device *device) override;
      
      OWLGeom const geom;
    };

    struct TrianglesGeom : public Geom {
      TrianglesGeom(OWLGeom geom);
      
      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(rtc::Buffer *vertices, int numVertices) override;
      void setIndices(rtc::Buffer *indices, int numIndices) override;
    };
    
    struct GeomType : public rtc::GeomType {
      GeomType(const DevGroup *devGroup) : devGroup(devGroup) {}
      ~GeomType() override;
      
      const DevGroup *const devGroup;
      OWLGeomType gt = 0;
    };
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(const DevGroup *dg,
                        const std::string &typeName,
                        size_t sizeOfDD,
                        const std::string &ahFctName,
                        const std::string &chFctName);
      rtc::Geom *createGeom() override;
    };
    
    struct DevGroup : public cuda::BaseDevGroup {
      DevGroup(OptixBackend *backend,
               const std::vector<int> &gpuIDs);
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
                              const char *chFctName) override;
      

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

      // ==================================================================
      // buffer stuff
      // ==================================================================
      rtc::Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0) const override;

      void free(rtc::Buffer *) const override;
      void copy(rtc::Buffer *dst,
                rtc::Buffer *src,
                size_t numBytes) const override;
         
      OWLContext      owl = 0;
    };
    
    struct OptixBackend : public cuda::BaseBackend {
      // setActive and getActive: inherited from cuda bacekdn
      OptixBackend();
      virtual ~OptixBackend() = default;
      
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs) override;
    };
    
  }
}

  
  
