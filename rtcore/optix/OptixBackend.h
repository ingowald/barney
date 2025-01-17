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

      rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms)
        override
      { BARNEY_NYI(); };
      
      rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) 
        override
      { BARNEY_NYI(); };

      void free(rtc::Group *)
        override 
      { BARNEY_NYI(); };
      
      std::vector<OptixDevice *> devices;
    };
    
    struct OptixBackend : public cuda::BaseBackend {
      // setActive and getActive: inherited from cuda bacekdn
      OptixBackend();
      
      rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                    size_t sizeOfGlobals) override;
    };
    
  }
}

  
  
