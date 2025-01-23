#include "rtcore/optix/OptixBackend.h"

extern "C" char traceRays_ptx[];

namespace barney {
  namespace optix {
    
    Device::Device(int physicalGPU)
      : cuda::BaseDevice(physicalGPU)
    {
      owl = owlContextCreate(&physicalGPU,1);
    }

    Device::~Device()
    {
      destroy();
    }

    void Device::destroy()
    {
      if (owl) owlContextDestroy(owl);
      owl = 0;
    }


    
    OptixBackend::OptixBackend()
    {}

    std::vector<rtc::Device *>
    OptixBackend::createDevices(const std::vector<int> &gpuIDs)
    {
      std::vector<rtc::Device *> devs;
      for (auto gpuID : gpuIDs)
        devs.push_back(new optix::Device(gpuID));
      return devs;
    }
    

    // ==================================================================
    // rtcore/pipeline stuff
    // ==================================================================
    void Device::buildPipeline() 
    {
      owlBuildPipeline(owl);
    }
      
    void Device::buildSBT() 
    {
      owlBuildSBT(owl);
    }
      
    

    // ==================================================================
    // buffer
    // ==================================================================

    void Device::freeBuffer(rtc::Buffer *buffer) 
    {
      delete buffer;
    }
    
    rtc::Buffer *Device::createBuffer(size_t numBytes,
                                      const void *initValues) 
    {
      return new Buffer(this,numBytes,initValues);
    }
    
    Buffer::Buffer(optix::Device *device,
                   size_t size,
                   const void *initData)
      : rtc::Buffer(device)
    {
      owl = owlDeviceBufferCreate(device->owl,OWL_BYTE,size,initData);
    }
    
    void *Buffer::getDD() const
    { return (void*)owlBufferGetPointer(owl,0); }
    
    // ==================================================================
    // geom
    // ==================================================================

    Geom::Geom(GeomType *gt,
               OWLGeom geom)
      : rtc::Geom(gt->device),
        owl(geom)
    {}
    
    Geom::~Geom()
    {
      owlGeomRelease(owl);
    }

    void Geom::setDD(const void *dd)
    {
      owlGeomSetRaw(owl,"raw",dd);
    }

    TrianglesGeom::TrianglesGeom(GeomType *gt,
                                 OWLGeom geom)
      : Geom(gt,geom)
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
      owlTrianglesSetVertices(owl,((Buffer*)vertices)->owl,
                              numVertices,sizeof(float3),0);
    }
    
    void TrianglesGeom::setIndices(rtc::Buffer *indices, int numIndices)
    {
      owlTrianglesSetIndices(owl,((Buffer*)indices)->owl,
                             numIndices,sizeof(int3),0);
    }
    
    // ==================================================================
    // geomtype
    // ==================================================================
    rtc::Geom *TrianglesGeomType::createGeom()
    {
      OWLGeom geom = owlGeomCreate(((optix::Device*)device)->owl,this->gt);
      return new TrianglesGeom(this,geom);
    }
    
    rtc::GeomType *
    Device::createTrianglesGeomType(const char *typeName,
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
    
    TrianglesGeomType::TrianglesGeomType(optix::Device *device,
                                         const std::string &typeName,
                                         size_t sizeOfDD,
                                         const std::string &ahFctName,
                                         const std::string &chFctName)
      : GeomType(device)
    {
      OWLVarDecl vars[] = {
        {"raw",(OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfDD),0},
        {nullptr}
      };
      gt = owlGeomTypeCreate(device->owl,OWL_GEOM_TRIANGLES,
                             sizeOfDD,vars,-1);
      
      const char *Triangles_ptx
        = (const char *)rtc::getSymbol(typeName+"_ptx");
      OWLModule module = owlModuleCreate
        (device->owl,Triangles_ptx);
      owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                               chFctName.c_str());//"TrianglesCH");
      owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,
                           ahFctName.c_str());//"TrianglesAH");
      owlBuildPrograms(device->owl);
      owlModuleRelease(module);
    }


    // ==================================================================
    // kernels
    // ==================================================================

    struct ComputeWrapper : public ::barney::rtc::Compute {
      typedef void (*LaunchFct)(vec3i, vec3i, int,
                                cudaStream_t, const void *);
      
      ComputeWrapper(Device *device,
                     const std::string &kernelName)
        : rtc::Compute(device)
      {
        const std::string symbolName
          = "barney_rtc_cuda_launch_"+kernelName;
        launchFct = (LaunchFct)rtc::getSymbol(symbolName);
      }
      
      LaunchFct launchFct = 0;
      
      void launch(int numBlocks,
                  int blockSize,
                  const void *dd) override
      {
        do_launch(vec3i(numBlocks,1,1),vec3i(blockSize,1,1),dd);
      }
      void launch(vec2i nb,
                  vec2i bs,
                  const void *dd) override
      {
        do_launch(vec3i(nb.x,nb.y,1),vec3i(bs.x,bs.y,1),dd);
      }
      void launch(vec3i nb,
                  vec3i bs,
                  const void *dd) override
      {
        do_launch(nb,bs,dd);
      }
      void do_launch(vec3i nb,
                     vec3i bs,
                     const void *dd) 
      {
        cuda::SetActiveGPU forDuration(device);
        launchFct(nb,bs,0,((cuda::BaseDevice*)device)->stream,dd);
      }
    };
    
    rtc::Compute *
    Device::createCompute(const std::string &kernelName) 
    {
      return new ComputeWrapper(this,kernelName);
    }


    struct TraceWrapper : public rtc::Trace
    {
      TraceWrapper(optix::Device *device,
                   const std::string &kernelName,
                   size_t sizeOfLP)
        : Trace(device)
      {
        const char *ptxCode = (const char *)
          rtc::getSymbol(kernelName+"_ptx");
        OWLVarDecl rg_args[]
          = {
          { nullptr }
        };
        mod = owlModuleCreate(device->owl,ptxCode);
        rg = owlRayGenCreate(device->owl,mod,
                             kernelName.c_str(),
                             0,rg_args,-1);
        owlBuildPrograms(device->owl);

        OWLVarDecl lp_args[]
          = {
          { "raw", (OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfLP), 0 },
          { nullptr }
        };
        lp = owlParamsCreate(device->owl,sizeOfLP,lp_args,-1);
      }

      void launch(vec2i dims, const void *dd) override
      {
        owlParamsSetRaw(lp,"raw",dd,0);
        owlAsyncLaunch2DOnDevice(rg,dims.x,dims.y,0,lp);
      }
      
      void launch(int launchDims,
                  const void *dd) override
      {
        BARNEY_NYI();
      }
      void sync() override
      {
        cudaStream_t s = owlParamsGetCudaStream(lp,0);
        BARNEY_CUDA_CALL(StreamSynchronize(s));
      }
     
      OWLModule mod;
      OWLRayGen rg;
      OWLLaunchParams lp;
    };
    
    rtc::Trace *
    Device::createTrace(const std::string &kernelName,
                          size_t sizeOfLP)
    {
      return new TraceWrapper(this,kernelName,sizeOfLP);
    }

    
    // ==================================================================
    // groups
    // ==================================================================
    void Device::freeGroup(rtc::Group *group)
    {
      delete group;
    }

    Group::Group(optix::Device *device, OWLGroup owl)
      : rtc::Group(device),
        owl(owl)
    {}
    
    rtc::Group *
    Device::createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->owl);
      OWLGroup g = owlTrianglesGeomGroupCreate(owl,
                                               owlGeoms.size(),
                                               owlGeoms.data());
      return new Group(this,g);
    }

    rtc::Group *
    Device::createInstanceGroup(const std::vector<rtc::Group *> &groups,
                                  const std::vector<affine3f> &xfms)
    {
      std::vector<OWLGroup> owls;
      for (auto group : groups)
        owls.push_back(((Group *)group)->owl);
      OWLGroup g
        = owlInstanceGroupCreate(owl,
                                 owls.size(),
                                 owls.data(),
                                 nullptr,
                                 (const float *)xfms.data());
      return new Group(this,g);
    }

    
    rtc::device::AccelHandle Group::getDD() const
    {
      OptixTraversableHandle handle
        = owlGroupGetTraversable(owl,0);
      return (const rtc::device::AccelHandle &)handle;
    }
    
    void Group::buildAccel()
    {
      owlGroupBuildAccel(owl);
    }
    
    void Group::refitAccel() 
    {
      owlGroupRefitAccel(owl);
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

  
  
