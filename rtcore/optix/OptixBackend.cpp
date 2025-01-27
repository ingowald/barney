#include "rtcore/optix/OptixBackend.h"

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

    rtc::Device *OptixBackend::createDevice(int gpuID) 
    {
      return new optix::Device(gpuID);
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
    UserGeom::UserGeom(GeomType *gt,
                                 OWLGeom geom)
      : Geom(gt,geom)
    {}
    
    /*! only for user geoms */
    void TrianglesGeom::setPrimCount(int primCount)
    {
      throw std::runtime_error
        ("invalid to call setprimcount on triangles");
    }
    /*! only for user geoms */
    void UserGeom::setPrimCount(int primCount)
    {
      owlGeomSetPrimCount(owl,primCount);
    }
    
    /*! can only get called on triangle type geoms */
    void TrianglesGeom::setVertices(rtc::Buffer *vertices, int numVertices)
    {
      owlTrianglesSetVertices(owl,((Buffer*)vertices)->owl,
                              numVertices,sizeof(float3),0);
    }
    /*! can only get called on triangle type geoms */
    void UserGeom::setVertices(rtc::Buffer *vertices, int numVertices)
    {
      /* ignore */
    }
    
    void TrianglesGeom::setIndices(rtc::Buffer *indices, int numIndices)
    {
      owlTrianglesSetIndices(owl,((Buffer*)indices)->owl,
                             numIndices,sizeof(int3),0);
    }
    /*! can only get called on triangle type geoms */
    void UserGeom::setIndices(rtc::Buffer *indices, int numIndices)
    {
      /* ignore */
    }
    
    // ==================================================================
    // geomtype
    // ==================================================================
    GeomType::~GeomType()
    {
      // CANNOYT yet release this because owl cannot do that yet
      gt = 0;
    }
    
    rtc::GeomType *
    Device::createTrianglesGeomType(const char *typeName,
                                    size_t sizeOfDD,
                                    bool has_ah,
                                    bool has_ch)
    {
      return new TrianglesGeomType(this,typeName,sizeOfDD,has_ah,has_ch);
    }
    
    rtc::GeomType *
    Device::createUserGeomType(const char *typeName,
                                    size_t sizeOfDD,
                                    bool has_ah,
                                    bool has_ch)
    {
      return new UserGeomType(this,typeName,sizeOfDD,has_ah,has_ch);
    }
    
    
    TrianglesGeomType::TrianglesGeomType(optix::Device *device,
                                         const std::string &typeName,
                                         size_t sizeOfDD,
                                         bool has_ah,
                                         bool has_ch)
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
      if (has_ch)
        owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                                 typeName.c_str());
                                 // chFctName.c_str());//"TrianglesCH");
      if (has_ah)
        owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,
                             typeName.c_str());
      // ahFctName.c_str());//"TrianglesAH");
      owlBuildPrograms(device->owl);
      owlModuleRelease(module);
    }

    UserGeomType::UserGeomType(optix::Device *device,
                               const std::string &typeName,
                               size_t sizeOfDD,
                               bool has_ah,
                               bool has_ch)
      : GeomType(device)
    {
      OWLVarDecl vars[] = {
        {"raw",(OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfDD),0},
        {nullptr}
      };
      gt = owlGeomTypeCreate(device->owl,OWL_GEOM_USER,
                             sizeOfDD,vars,-1);
      
      const char *User_ptx
        = (const char *)rtc::getSymbol(typeName+"_ptx");
      OWLModule module = owlModuleCreate
        (device->owl,User_ptx);
      if (has_ch)
        owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                                 typeName.c_str());
      if (has_ah)
        owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,
                             typeName.c_str());
      owlGeomTypeSetBoundsProg(gt,/*ray type*/module,
                               typeName.c_str());
      owlGeomTypeSetIntersectProg(gt,/*ray type*/0,module,
                                  typeName.c_str());
      owlBuildPrograms(device->owl);
      owlModuleRelease(module);
    }
    
    rtc::Geom *TrianglesGeomType::createGeom()
    {
      OWLGeom geom = owlGeomCreate(((optix::Device*)device)->owl,this->gt);
      return new TrianglesGeom(this,geom);
    }

    rtc::Geom *UserGeomType::createGeom()
    {
      OWLGeom geom = owlGeomCreate(((optix::Device*)device)->owl,this->gt);
      return new UserGeom(this,geom);
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

        cudaDeviceSynchronize();
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
        : Trace(device),
          device(device)
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
        lpStream = owlParamsGetCudaStream(lp,0);
      }

      void launch(vec2i dims, const void *dd) override
      {
        // 'in theory' we should be able to share the stream between
        // cuda kernels and trace kernels; but at least right now we
        // can't tell owl which stream to use, so owl generates its
        // own stream (per lp, to boot). so we can't currently
        // schedule this trace call into the existing cuda stream, and
        // consequently have to do a manual sync here to make sure
        // that whatever is in the cuda stream is already done by the
        // time the owl trace gets launched 
        cuda::SetActiveGPU forDuration(device);
        BARNEY_CUDA_CALL(StreamSynchronize(/*inherited!*/device->stream));
        
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
        cuda::SetActiveGPU forDuration(device);
        BARNEY_CUDA_CALL(StreamSynchronize(lpStream));
        // cudaDeviceSynchronize();
      }
     
      OWLModule mod;
      OWLRayGen rg;
      OWLLaunchParams lp;
      cudaStream_t lpStream;
      optix::Device *const device;
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
    Device::createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->owl);
      OWLGroup g = owlUserGeomGroupCreate(owl,
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

    __attribute__((visibility("default")))
    Backend *createBackend_optix()
    {
      PING;
      Backend *be = new barney::optix::OptixBackend;

      PRINT(be);
      return be;
    }
  }
}

  
  
