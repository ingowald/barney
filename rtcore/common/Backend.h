#pragma once

#include "barney/common/barney-common.h"

namespace barney {
  namespace rtc {
    struct device {
      typedef struct _OpaqueAccel   *AccelHandle;
      typedef struct _OpaqueTextureObject *TextureObject;
    };
    
    struct Backend;
    struct Device;
    struct Geom;
    struct Group;

    struct Group {
      virtual rtc::device::AccelHandle getDD(Device *) const = 0;
      
      virtual void buildAccel() = 0;
      virtual void refitAccel() = 0;
    };

    struct Buffer {
      virtual void *getDD(Device *) const = 0;
      virtual void upload(const void *hostPtr,
                          size_t numBytes,
                          size_t ofs = 0) = 0;
    };
    struct Texture {
      const vec3i &getDims() const;
      virtual void *getDD(Device *) const = 0;
    };
    
    struct Device {
      Device(Backend *const backend,
             const int physicalID)
        : backend(backend),
          physicalID(physicalID)
      {}

      virtual void launchTrace(const void *ddPtr) = 0;
      
      Backend *const backend;
      const int physicalID;
    };

    struct DevGroup {
      DevGroup(Backend *backend)
        : backend(backend)
      {}

      // ==================================================================
      // buffer stuff
      // ==================================================================
      virtual Buffer *createBuffer(size_t numBytes) const
      { BARNEY_NYI(); }
      virtual void free(Buffer *) const
      { BARNEY_NYI(); }
      virtual void copy(Buffer *dst, Buffer *src, size_t numBytes) const
      { BARNEY_NYI(); }
      
      // ==================================================================
      // group/accel stuff
      // ==================================================================
      virtual Group *
      createTrianglesGroup(const std::vector<Geom *> &geoms) = 0;
      
      virtual Group *
      createUserGeomsGroup(const std::vector<Geom *> &geoms) = 0;

      virtual Group *
      createInstanceGroup(const std::vector<Group *> &groups,
                          const std::vector<affine3f> &xfms)
      { BARNEY_NYI(); }
      
      virtual void free(rtc::Group *) = 0;

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

    // helper function(s)
    template<typename T> void resizeAndUpload(rtc::Buffer *&buffer,
                                              const std::vector<T> &data);


    

    Backend *createBackend_cuda();
    Backend *createBackend_optix();
    Backend *createBackend_embree();
  }
}



// TODO:
template<typename T>
inline __device__ __host__ T tex2D(barney::rtc::device::TextureObject to,
                                   float x, float y)
{
#ifdef __CUDA_ARCH__
  printf("tex2d missing...\n");
  return T{};
#else
  BARNEY_NYI();
#endif
}
