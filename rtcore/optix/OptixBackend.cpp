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
    // rtcore/pipeline stuff
    // ==================================================================
    void DevGroup::buildPipeline() 
    {
      owlBuildPipeline(owl);
    }
      
    void DevGroup::buildSBT() 
    {
      owlBuildSBT(owl);
    }
      
    

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
        {"raw",(OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfDD),0},
        {nullptr}
      };
      gt = owlGeomTypeCreate(devGroup->owl,OWL_GEOM_TRIANGLES,
                             sizeOfDD,vars,-1);
      
      const char *Triangles_ptx
        = (const char *)rtc::getSymbol(typeName+"_ptx");
      OWLModule module = owlModuleCreate
        (devGroup->owl,Triangles_ptx);
      owlGeomTypeSetClosestHit(gt,/*ray type*/0,module,
                               chFctName.c_str());//"TrianglesCH");
      owlGeomTypeSetAnyHit(gt,/*ray type*/0,module,
                           ahFctName.c_str());//"TrianglesAH");
      owlBuildPrograms(devGroup->owl);
      owlModuleRelease(module);
    }


    // ==================================================================
    // kernels
    // ==================================================================

    struct ComputeWrapper : public ::barney::rtc::ComputeKernel {
      typedef void (*LaunchFct)(vec3i, vec3i, int, cudaStream_t, const void *);
      
      ComputeWrapper(const std::string &kernelName)        
      {
        const std::string symbolName
          = "barney_rtc_cuda_launch_"+kernelName;
        launchFct = (LaunchFct)rtc::getSymbol(symbolName);
      }
      
      LaunchFct launchFct = 0;
      
      void launch(rtc::Device *device,
                  int numBlocks,
                  int blockSize,
                  const void *dd) override
      {
        do_launch(device,vec3i(numBlocks,1,1),vec3i(blockSize,1,1),dd);
      }
      void launch(rtc::Device *device,
                  vec2i nb,
                  vec2i bs,
                  const void *dd) override
      {
        do_launch(device,vec3i(nb.x,nb.y,1),vec3i(bs.x,bs.y,1),dd);
      }
      void launch(rtc::Device *device,
                  vec3i nb,
                  vec3i bs,
                  const void *dd) override
      {
        do_launch(device,nb,bs,dd);
      }
      void do_launch(rtc::Device *device,
                     vec3i nb,
                     vec3i bs,
                     const void *dd) 
      {
        cuda::SetActiveGPU forDuration(device);
        launchFct(nb,bs,0,((cuda::BaseDevice*)device)->stream,dd);
      }
    };
    
    rtc::ComputeKernel *
    DevGroup::createCompute(const std::string &kernelName) 
    {
      return new ComputeWrapper(kernelName);
    }


    struct TraceWrapper : public rtc::TraceKernel
    {
      TraceWrapper(optix::DevGroup *dg,
                   const std::string &kernelName,
                   size_t sizeOfLP)
      {
        const char *ptxCode = (const char *)
          rtc::getSymbol(kernelName+"_ptx");
        OWLVarDecl rg_args[]
          = {
          { nullptr }
        };
        mod = owlModuleCreate(dg->owl,ptxCode);
        rg = owlRayGenCreate(dg->owl,mod,
                             kernelName.c_str(),
                             0,rg_args,-1);
        owlBuildPrograms(dg->owl);

        PRINT(sizeOfLP);
        OWLVarDecl lp_args[]
          = {
          { "raw", (OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfLP), 0 },
          { nullptr }
        };
        lp = owlParamsCreate(dg->owl,sizeOfLP,lp_args,-1);
      }

      void launch(rtc::Device *device,
                  vec2i dims,
                  const void *dd) override
      {
        int devID = ((OptixDevice*)device)->localID;
        owlParamsSetRaw(lp,"raw",dd,devID);
        owlAsyncLaunch2DOnDevice(rg,dims.x,dims.y,devID,lp);
      }
      void launch(rtc::Device *device,
                  int launchDims,
                  const void *dd) override
      {
        BARNEY_NYI();
      }
      void sync(rtc::Device *device) override
      {
        int devID = ((OptixDevice*)device)->localID;
        cudaStream_t s = owlParamsGetCudaStream(lp,devID);
        BARNEY_CUDA_CALL(StreamSynchronize(s));
      }
      
      
      OWLModule mod;
      OWLRayGen rg;
      OWLLaunchParams lp;
    };
    
    rtc::TraceKernel *
    DevGroup::createTrace(const std::string &kernelName,
                          size_t sizeOfLP)
    {
      return new TraceWrapper(this,kernelName,sizeOfLP);
    }

    
    // ==================================================================
    // groups
    // ==================================================================
    rtc::Group *
    DevGroup::createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->geom);
      OWLGroup g = owlTrianglesGeomGroupCreate(owl,
                                               owlGeoms.size(),
                                               owlGeoms.data());
      return new Group(g);
    }

    rtc::Group *
    DevGroup::createInstanceGroup(const std::vector<rtc::Group *> &groups,
                                  const std::vector<affine3f> &xfms)
    {
      std::vector<OWLGroup> owlGroups;
      for (auto group : groups)
        owlGroups.push_back(((Group *)group)->owlGroup);
      OWLGroup g
        = owlInstanceGroupCreate(owl,
                                 owlGroups.size(),
                                 owlGroups.data(),
                                 nullptr,
                                 (const float *)xfms.data());
      return new Group(g);
    }

    
    rtc::device::AccelHandle
    Group::getDD(const rtc::Device *device) const
    {
      OptixTraversableHandle handle
        = owlGroupGetTraversable(owlGroup,device->localID);
      return (const rtc::device::AccelHandle &)handle;
    }
    
    void Group::buildAccel()
    {
      owlGroupBuildAccel(owlGroup);
    }
    
    void Group::refitAccel() 
    {
      owlGroupRefitAccel(owlGroup);
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

  
  
