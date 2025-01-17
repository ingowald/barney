#pragma once

#include "barney/common/barney-common.h"

namespace barney {
  namespace rtc {

    typedef struct _OpaqueTextureHandle *OpaqueTextureHandle;

    struct Backend;
    
    struct Device {
      Device(Backend *const backend,
             const int physicalID)
        : backend(backend),
          physicalID(physicalID)
      {}
      
      Backend *const backend;
      const int physicalID;
    };

    struct DevGroup {
      DevGroup(Backend *backend)
        : backend(backend)
      {}
      
      Backend *const backend;
    };
    
    struct Backend {
      typedef std::shared_ptr<Backend> SP;
      virtual ~Backend() = default;

      virtual void setActiveGPU(int physicalID) = 0;
      virtual int  getActiveGPU() = 0;
      virtual DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                       size_t sizeOfGlobals) = 0;
      
      /*! number of 'physical' devices - num cuda capable gpus if cuda
        or optix, or 1 if embree */
      int numPhysicalDevices = 0;

      static int getDeviceCount();
      static rtc::Backend *get();
    private:
      static rtc::Backend *create();
      static rtc::Backend *singleton;
    };


    Backend *createBackend_cuda();
    Backend *createBackend_optix();
    Backend *createBackend_embree();
  }
}
