#include "rtcore/optix/OptixBackend.h"

extern "C" char traceRays_ptx[];

namespace barney {
  namespace optix {

    OptixDevice::OptixDevice(DevGroup *parent,
                             int physicalGPU,
                             size_t sizeOfGlobals)
      : parent(parent), physicalGPU(physicalGPU)
    {
      owl = owlContextCreate((int*)&physicalGPU,1);
      OWLVarDecl args[]
        = {
        { nullptr }
      };
      OWLModule module = owlModuleCreate(owl,traceRays_ptx);
      rg = owlRayGenCreate(owl,module,"traceRays",0,args,-1);
      
      owlBuildPrograms(owl);

      OWLVarDecl params[]
        = {
        { "raw", OWL_USER_TYPE(sizeOfGlobals), 0 },
        { nullptr }
      };
      lp = owlParamsCreate(owl,
                           sizeOfGlobals,
                           params,
                           -1);
    }
    
    DevGroup::DevGroup(OptixBackend *backend,
                       const std::vector<int> &gpuIDs,
                       size_t sizeOfGlobals)
      : cuda::BaseDevGroup(backend,gpuIDs,sizeOfGlobals)
    {

      for (int devID=0;devID<gpuIDs.size();devID++)
        devices.push_back(new OptixDevice(this,gpuIDs[devID],sizeOfGlobals));
    }
    
    rtc::DevGroup *OptixBackend
    ::createDevGroup(const std::vector<int> &gpuIDs,
                     size_t sizeOfGlobals)
    {
      return new DevGroup(this,gpuIDs,sizeOfGlobals);
    }
    
    OptixBackend::OptixBackend()
    {}
  }
  namespace rtc {
    
    Backend *createBackend_optix()
    {
      PING;
      return new barney::optix::OptixBackend;
    }
  }
}

  
  
