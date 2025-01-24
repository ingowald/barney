#pragma once

#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace optix {
    struct OptixBackend;

    struct Device : public cuda::BaseDevice {
      Device(int physicalGPU);
      virtual ~Device();

      void destroy() override;
      
      // ==================================================================
      // kernels
      // ==================================================================
      rtc::Compute *
      createCompute(const std::string &) override;
      
      rtc::Trace *
      createTrace(const std::string &, size_t) override;

      // ==================================================================
      // buffer stuff
      // ==================================================================
      rtc::Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0) override;
      void freeBuffer(rtc::Buffer *) override;
      
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

      void buildPipeline() override;
      void buildSBT() override;

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
      { BARNEY_NYI(); };
      
      rtc::GeomType *
      createTrianglesGeomType(const char *typeName,
                              size_t sizeOfDD,
                              bool has_ah,
                              bool has_ch) override;
      
      void freeGeomType(rtc::GeomType *) override 
      { BARNEY_NYI(); };

      // ------------------------------------------------------------------
      // geom stuff
      // ------------------------------------------------------------------
      
      void freeGeom(rtc::Geom *) override 
      { BARNEY_NYI(); };
      
      // ------------------------------------------------------------------
      // group/accel stuff
      // ------------------------------------------------------------------
      rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms) override;
      
      rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) override
      { BARNEY_NYI(); };

      rtc::Group *
      createInstanceGroup(const std::vector<rtc::Group *> &groups,
                          const std::vector<affine3f> &xfms) override;

      void freeGroup(rtc::Group *) override;
         
      OWLContext      owl = 0;
    };
    
    
    struct Buffer : public rtc::Buffer {
      Buffer(optix::Device *device,
             size_t size,
             const void *initData);
      void *getDD() const override;
      
      OWLBuffer owl;
    };

    struct Group : public rtc::Group {
      Group(optix::Device *device, OWLGroup owlGroup);
      virtual ~Group() { owlGroupRelease(owl); }
      
      rtc::device::AccelHandle getDD() const override;
      void buildAccel() override;
      void refitAccel() override;
      
      OWLGroup const owl;
    };

    struct GeomType;
    
    struct Geom : public rtc::Geom {
      Geom(GeomType *gt, OWLGeom geom);
      virtual ~Geom();
      void setDD(const void *dd) override;
      
      OWLGeom const owl;
    };

    struct TrianglesGeom : public Geom {
      TrianglesGeom(GeomType *gt, OWLGeom geom);
      
      /*! only for user geoms */
      void setPrimCount(int primCount) override;
      /*! can only get called on triangle type geoms */
      void setVertices(rtc::Buffer *vertices, int numVertices) override;
      void setIndices(rtc::Buffer *indices, int numIndices) override;
    };
    
    struct GeomType : public rtc::GeomType {
      GeomType(optix::Device *device) : rtc::GeomType(device) {}
      ~GeomType() override;
      
      OWLGeomType gt = 0;
    };
    struct TrianglesGeomType : public GeomType
    {
      TrianglesGeomType(optix::Device *device,
                        const std::string &typeName,
                        size_t sizeOfDD, bool has_ah, bool has_ch);
      rtc::Geom *createGeom() override;
    };
    
    struct OptixBackend : public cuda::BaseBackend {
      // setActive and getActive: inherited from cuda bacekdn
      OptixBackend();
      virtual ~OptixBackend() = default;
      
      rtc::Device *createDevice(int gpuID) override;
    };
    
  }
}

  
  
