#include "rtcore/optix/OptixBackend.h"

extern "C" char traceRays_ptx[];

namespace barney {
  namespace optix {
    
    OptixDevice::OptixDevice(const optix::DevGroup *parent,
                             int physicalGPU,
                             int localID)
      : cuda::BaseDevice(physicalGPU,localID),
        parent(parent)
    {
      // owl = owlContextCreate((int*)&physicalGPU,1);
      // OWLVarDecl args[]
      //   = {
      //   { nullptr }
      // };
      // OWLModule module = owlModuleCreate(owl,traceRays_ptx);
      // rg = owlRayGenCreate(owl,module,"traceRays",0,args,-1);
      
      // owlBuildPrograms(owl);

      // OWLVarDecl params[]
      //   = {
      //   { "raw", OWL_USER_TYPE(sizeOfGlobals), 0 },
      //   { nullptr },,
      // };
      // lp = owlParamsCreate(owl,
      //                      sizeOfGlobals,
      //                      params,
      //                      -1);
    }

    OptixDevice::~OptixDevice()
    {
    }

    
    DevGroup::DevGroup(OptixBackend *backend,
                       const std::vector<int> &gpuIDs)
      : cuda::BaseDevGroup(backend,gpuIDs)
    {
      owl = owlContextCreate((int*)gpuIDs.data(),gpuIDs.size());

      for (int owlID=0;owlID<gpuIDs.size();owlID++)
        devices.push_back(new OptixDevice(this,gpuIDs[owlID],owlID));
    }

    DevGroup::~DevGroup()
    {
      for (auto device : devices)
        delete device;
      devices.clear();
      owlContextDestroy(owl);
      owl = 0;
    }
    
    rtc::DevGroup *OptixBackend
    ::createDevGroup(const std::vector<int> &gpuIDs)
    {
      assert(!gpuIDs.empty());
      optix::DevGroup *dg = new DevGroup(this,gpuIDs);
      // assert(dg);
      // for (auto gpuID : gpuIDs)
      //   dg->devices.push_back(new OptixDevice(dg,gpuID));
      return dg;
    }
    
    OptixBackend::OptixBackend()
    {}


    // ==================================================================
    // buffer
    // ==================================================================

    void DevGroup::free(rtc::Buffer *buffer) const
    {
      delete buffer;
    }
    
    void DevGroup::copy(rtc::Buffer *dst,
                        rtc::Buffer *src,
                        size_t numBytes) const
    {
      for (auto dev : devices) {
        BARNEY_CUDA_CALL(Memcpy(dst->getDD(dev),src->getDD(dev),
                                numBytes,cudaMemcpyDefault));
      }
    }
    
    rtc::Buffer *DevGroup::createBuffer(size_t numBytes,
                                        const void *initValues) const
    {
      return new Buffer(this,numBytes,initValues);
    }

    Buffer::Buffer(const optix::DevGroup *dg,
                             size_t size,
                             const void *initData)
    {
      owl = owlDeviceBufferCreate(dg->owl,OWL_BYTE,size,initData);
    }
    
    void *Buffer::getDD(const rtc::Device *device) const
    { return (void*)owlBufferGetPointer(owl,device->localID); }
    
    void Buffer::upload(const void *hostPtr,
                             size_t numBytes,
                             size_t ofs,
                             const rtc::Device *device)
    {
      if (device) {
        uint8_t *devPtr = (uint8_t*)owlBufferGetPointer(owl,device->localID);
        BARNEY_CUDA_CALL(Memcpy(devPtr+ofs,hostPtr,
                                numBytes,cudaMemcpyDefault));
      } else {
        owlBufferUpload(owl,hostPtr,ofs,numBytes);
      }
    }
    

    // ==================================================================
    // geom
    // ==================================================================

    Geom::Geom(OWLGeom geom)
      : geom(geom)
    {}
    
    Geom::~Geom()
    {
      owlGeomRelease(geom);
    }

    void Geom::setDD(const void *dd, const rtc::Device *device)
    {
      owlGeomSetRaw(geom,"raw",dd);
    }

    TrianglesGeom::TrianglesGeom(OWLGeom geom)
      : Geom(geom)
    {}
    
    /*! only for user geoms */
    void TrianglesGeom::setPrimCount(int primCount)
    {
      throw std::runtime_error
        ("invalid to call setprimcount on triangles");
    }
    
    /*! can only get called on triangle type geoms */
    void TrianglesGeom::setVertices(rtc::Buffer *vertices, int numVertices)
    {
      owlTrianglesSetVertices(geom,((Buffer*)vertices)->owl,
                              numVertices,sizeof(float3),0);
    }
    
    void TrianglesGeom::setIndices(rtc::Buffer *indices, int numIndices)
    {
      owlTrianglesSetIndices(geom,((Buffer*)indices)->owl,
                             numIndices,sizeof(int3),0);
    }
    
    // ==================================================================
    // geomtype
    // ==================================================================
    rtc::Geom *TrianglesGeomType::createGeom()
    {
      OWLGeom geom = owlGeomCreate(devGroup->owl,gt);
      return new TrianglesGeom(geom);
    }
    
    rtc::GeomType *
    DevGroup::createTrianglesGeomType(const char *typeName,
                                      size_t sizeOfDD,
                                      const char *ahFctName,
                                      const char *chFctName)
    {
      return new TrianglesGeomType(this,typeName,sizeOfDD,
                                   ahFctName,chFctName);
    }

    GeomType::~GeomType()
    {
      //      owlGeomTypeRelease(gt);
      gt = 0;
    }
    
    TrianglesGeomType::TrianglesGeomType(const DevGroup *devGroup,
                                         const std::string &typeName,
                                         size_t sizeOfDD,
                                         const std::string &ahFctName,
                                         const std::string &chFctName)
      : GeomType(devGroup)
    {
      OWLVarDecl vars[] = {
        {"raw",OWL_USER_TYPE(sizeOfDD),0},
        {nullptr}
      };
      gt = owlGeomTypeCreate(devGroup->owl,OWL_GEOM_TRIANGLES,
                             sizeOfDD,vars,-1);
      
      const char *Triangles_ptx
        = (const char *)rtc::getSymbol(typeName+"_ptx");
      OWLModule module = owlModuleCreate
        (devGroup->owl,Triangles_ptx);
      owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,"TrianglesCH");
      owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,"TrianglesAH");
      owlBuildPrograms(devGroup->owl);
      owlModuleRelease(module);
    }
    
  }
  namespace rtc {
    
    Backend *createBackend_optix()
    {
      PING;
      return new barney::optix::OptixBackend;
    }
  }
}

  
  
