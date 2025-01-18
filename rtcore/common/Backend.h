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
    struct GeomType;
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
                          size_t ofs = 0,
                          const Device *device=nullptr) = 0;
    };

    struct TextureData {
      typedef enum {
        FLOAT,
        FLOAT4,
        UCHAR,
        UCHAR4,
        USHORT,
      } Format;
    };
    
    struct Texture {
      typedef enum {
        WRAP,CLAMP,BORDER,MIRROR,
      } AddressMode;
      
      typedef enum {
        FILTER_MODE_NEAREST,FILTER_MODE_LINEAR,
      } FilterMode;
    
      typedef enum {
        COLOR_SPACE_LINEAR, COLOR_SPACE_SRGB,
      } ColorSpace;
    
      const vec3i &getDims() const;
      virtual rtc::device::TextureObject getDD(Device *) const = 0;
    };
    
    struct Device {
      Device(Backend *const backend,
             const int physicalID)
        : backend(backend),
          physicalID(physicalID)
      {}

      virtual void copyAsync(void *dst, void *src, size_t numBytes)
      { BARNEY_NYI(); }
      
      virtual void *alloc(size_t numBytes)
      { BARNEY_NYI(); }
      
      virtual void free(void *mem)
      { BARNEY_NYI(); }
      
      virtual void buildPipeline()
      { BARNEY_NYI(); }
      
      virtual void buildSBT()
      { BARNEY_NYI(); }
      
      /*! sets this gpu as active, and returns physical ID of GPU that
          was active before */
      virtual int setActive() const = 0;
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      virtual void restoreActive(int oldActive) const = 0;
      
      // virtual void launchTrace(const void *ddPtr) = 0;
      virtual void sync() { BARNEY_NYI(); }
      
      Backend *const backend;
      const int physicalID;
    };


    struct ComputeKernel {
      virtual void launch(rtc::Device *device,
                          vec2i numBlocks,
                          vec2i blockSize,
                          const void *dd)
      { BARNEY_NYI(); }
      virtual void launch(rtc::Device *device,
                          int numBlocks,
                          int blockSize,
                          const void *dd)
      { BARNEY_NYI(); }
      virtual void sync(rtc::Device *device)
      { BARNEY_NYI(); }
    };

    struct TraceKernel {
      virtual void launch(rtc::Device *device,
                          vec2i launchDims,
                          const void *dd)
      { BARNEY_NYI(); }
      virtual void launch(rtc::Device *device,
                          int launchDims,
                          const void *dd)
      { BARNEY_NYI(); }
      virtual void sync(rtc::Device *device)
      { BARNEY_NYI(); }
    };
    
    struct DevGroup {
      DevGroup(Backend *backend)
        : backend(backend)
      {}
      virtual ~DevGroup() {}

      virtual void destroy()
      { BARNEY_NYI(); }
      
      // ==================================================================
      // kernels
      // ==================================================================
      rtc::ComputeKernel *createCompute(size_t ddSize);
      rtc::TraceKernel *createTrace(size_t ddSize);
      
      // ==================================================================
      // buffer stuff
      // ==================================================================
      virtual Buffer *createBuffer(size_t numBytes, const void *initValues = 0) const
      { BARNEY_NYI(); }
      virtual void free(Buffer *) const
      { BARNEY_NYI(); }
      virtual void copy(Buffer *dst, Buffer *src, size_t numBytes) const
      { BARNEY_NYI(); }
      
      // ==================================================================
      // texture stuff
      // ==================================================================

      virtual rtc::TextureData *
      createTextureData(vec3i dims,
                        rtc::TextureData::Format format,
                        const void *texels) const
      { BARNEY_NYI(); }
                        
      virtual rtc::Texture *
      createTexture(rtc::TextureData *data,
                    rtc::Texture::FilterMode filterMode,
                    rtc::Texture::AddressMode addressModes[3],
                    const vec4f borderColor = vec4f(0.f),
                    rtc::Texture::ColorSpace colorSpace
                    = rtc::Texture::COLOR_SPACE_LINEAR) const
      { BARNEY_NYI(); }
                        
      virtual void free(TextureData *) const
      { BARNEY_NYI(); }
      virtual void free(Texture *) const
      { BARNEY_NYI(); }

      // ==================================================================
      // geom/geomtype stuff
      // ==================================================================

      virtual rtc::GeomType *createGeomType(const char *typeName,
                                            size_t sizeOfDD,
                                            const char *boundsFctName,
                                            const char *isecFctName,
                                            const char *ahFctName,
                                            const char *chFctName)
      { BARNEY_NYI(); }
      
      virtual void free(Geom *)
      { BARNEY_NYI(); }
      
      virtual void free(GeomType *)
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

      std::vector<Device *> devices;
      Backend *const backend;
    };
    
    struct Backend {
      typedef std::shared_ptr<Backend> SP;
      virtual ~Backend() = default;

      // virtual void setActiveGPU(int physicalID) = 0;
      // virtual int  getActiveGPU() = 0;
      virtual rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs,
                                       size_t sizeOfGlobals) = 0;

      virtual rtc::Device *createDevice(int physicalID);

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
inline __device__ __host__ T tex1D(barney::rtc::device::TextureObject to,
                                   float x)
{
#ifdef __CUDA_ARCH__
  printf("tex2d missing...\n");
  return T{};
#else
  BARNEY_NYI();
#endif
}
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
template<typename T>
inline __device__ __host__ T tex3D(barney::rtc::device::TextureObject to,
                                   float x, float y, float z)
{
#ifdef __CUDA_ARCH__
  printf("tex2d missing...\n");
  return T{};
#else
  BARNEY_NYI();
#endif
}
