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
    
    rtc::Buffer *DevGroup::createBuffer(size_t numBytes,
                                        const void *initValues) const
    {
      return new OptixBuffer(this,numBytes,initValues);
    }

    OptixBuffer::OptixBuffer(const optix::DevGroup *dg,
                             size_t size,
                             const void *initData)
    {
      owl = owlDeviceBufferCreate(dg->owl,OWL_BYTE,size,initData);
    }
    
    void *OptixBuffer::getDD(const rtc::Device *device) const
    { return (void*)owlBufferGetPointer(owl,device->localID); }
    
    void OptixBuffer::upload(const void *hostPtr,
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
    

  }
  namespace rtc {
    
    Backend *createBackend_optix()
    {
      PING;
      return new barney::optix::OptixBackend;
    }
  }
}

  
  
