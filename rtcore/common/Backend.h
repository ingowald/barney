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

    const void *getSymbol(const std::string &symName);
    
    struct GeomType {
      virtual rtc::Geom *createGeom() = 0;
        
      virtual ~GeomType() = default;
    };
    
    struct Group {
      virtual rtc::device::AccelHandle
      getDD(const rtc::Device *) const = 0;
      
      virtual void buildAccel() = 0;
      virtual void refitAccel() = 0;
    };

    struct Buffer {
      virtual void *getDD(const rtc::Device *) const = 0;
      virtual void upload(const void *hostPtr,
                          size_t numBytes,
                          size_t ofs = 0,
                          const rtc::Device *device=nullptr) = 0;
    };

    struct Geom {
      /*! only for user geoms */
      virtual void setPrimCount(int primCount) = 0;
      /*! can only get called on triangle type geoms */
      virtual void setVertices(rtc::Buffer *vertices, int numVertices) = 0;
      virtual void setIndices(rtc::Buffer *indices, int numIndices) = 0;
      virtual void setDD(const void *dd, const rtc::Device *device) = 0;
    };

    struct TextureData {
      typedef enum {
        FLOAT,
        FLOAT4,
        UCHAR,
        UCHAR4,
        USHORT,
      } Format;
      TextureData(const vec3i dims,
                  const rtc::TextureData::Format format)
        : dims(dims), format(format)
      {}
        
      const vec3i dims;
      const Format format;
    };
    
    struct Texture {
      typedef enum {
        WRAP,CLAMP,BORDER,MIRROR,
      } AddressMode;
      
      typedef enum {
        FILTER_MODE_POINT,FILTER_MODE_LINEAR,
      } FilterMode;
    
      typedef enum {
        COLOR_SPACE_LINEAR, COLOR_SPACE_SRGB,
      } ColorSpace;

      // typedef enum {
      //   NORMALIZED_FLOAT, ELEMENT_TYPE
      // } ReadMode;

      Texture(TextureData *const data)
        : data(data)
      {}
      
      const vec3i &getDims() const {
        return data->dims;
      }
      virtual rtc::device::TextureObject getDD(const rtc::Device *) const = 0;

      TextureData *const data;
    };
    
    struct ComputeKernel {
      virtual void launch(rtc::Device *device,
                          int numBlocks,
                          int blockSize,
                          const void *dd)
      { BARNEY_NYI(); }

      virtual void launch(rtc::Device *device,
                          vec2i numBlocks,
                          vec2i blockSize,
                          const void *dd)
      { BARNEY_NYI(); }

      virtual void launch(rtc::Device *device,
                          vec3i numBlocks,
                          vec3i blockSize,
                          const void *dd)
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
      virtual void sync()
      { BARNEY_NYI(); }
    };
    


    struct Device {
      Device(const int physicalID, int localID)
        : physicalID(physicalID),
          localID(localID)
      {}

      virtual void *alloc(size_t numBytes) = 0;
      virtual void free(void *mem) = 0;
      virtual void *allocHost(size_t numBytes) = 0;
      virtual void freeHost(void *mem) = 0;
      virtual void memsetAsync(void *mem,int value, size_t size) = 0;
      virtual void copyAsync(void *dst, void *src, size_t size) = 0;
      virtual void buildPipeline() = 0;
      virtual void buildSBT() = 0;
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      virtual int setActive() const = 0;
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      virtual void restoreActive(int oldActive) const = 0;
      
      // virtual void launchTrace(const void *ddPtr) = 0;
      virtual void sync() = 0;
      
      // Backend *const backend;
      const int physicalID;
      const int localID;
    };


    struct DevGroup {
      DevGroup(Backend *backend)
        : backend(backend)
      {}
      virtual ~DevGroup() {}

      // /*! this must only ever get called by Backend::createDevGroup */
      // virtual rtc::Device *createDevice(int physicalID) = 0;
      
      virtual void destroy() = 0;
      
      // ==================================================================
      // kernels
      // ==================================================================
      virtual rtc::ComputeKernel *createCompute(const std::string &) = 0;
      
      virtual rtc::TraceKernel *createTrace(const std::string &) = 0;
      
      // ==================================================================
      // buffer stuff
      // ==================================================================
      virtual rtc::Buffer *createBuffer(size_t numBytes,
                                        const void *initValues = 0) const = 0;
      virtual void free(rtc::Buffer *) const = 0;
      virtual void copy(rtc::Buffer *dst,
                        rtc::Buffer *src,
                        size_t numBytes) const = 0;
      
      // ==================================================================
      // texture stuff
      // ==================================================================

      virtual rtc::TextureData *
      createTextureData(vec3i dims,
                        rtc::TextureData::Format format,
                        const void *texels) const = 0;
                        
      virtual rtc::Texture *
      createTexture(rtc::TextureData *data,
                    // rtc::Texture::ReadMode   readMode,
                    rtc::Texture::FilterMode filterMode,
                    rtc::Texture::AddressMode addressModes[3],
                    const vec4f borderColor = vec4f(0.f),
                    bool normalizedCoords = true,
                    rtc::Texture::ColorSpace colorSpace
                    = rtc::Texture::COLOR_SPACE_LINEAR) const = 0;
                        
      virtual void freeTextureData(rtc::TextureData *) const = 0;
      virtual void freeTexture(rtc::Texture *) const = 0;

      // ==================================================================
      // geom/geomtype stuff
      // ==================================================================

      virtual rtc::GeomType *
      createUserGeomType(const char *typeName,
                         size_t sizeOfDD,
                         const char *boundsFctName,
                         const char *isecFctName,
                         const char *ahFctName,
                         const char *chFctName) = 0;
      
      virtual rtc::GeomType *
      createTrianglesGeomType(const char *typeName,
                              size_t sizeOfDD,
                              const char *ahFctName,
                              const char *chFctName) = 0;
      
      virtual void free(rtc::Geom *) = 0;
      
      virtual void free(rtc::GeomType *) = 0;
      
      // ==================================================================
      // group/accel stuff
      // ==================================================================
      virtual rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms) = 0;
      
      virtual rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) = 0;

      virtual rtc::Group *
      createInstanceGroup(const std::vector<rtc::Group *> &groups,
                          const std::vector<affine3f> &xfms) = 0;
      
      virtual void free(rtc::Group *) = 0;

      std::vector<Device *> devices;
      Backend *const backend;
    };
    
    struct Backend {
      typedef std::shared_ptr<Backend> SP;
      virtual ~Backend() = default;

      // virtual void setActiveGPU(int physicalID) = 0;
      // virtual int  getActiveGPU() = 0;
      virtual rtc::DevGroup *createDevGroup(const std::vector<int> &gpuIDs
                                            //      ,
                                            // size_t sizeOfGlobals
                                            ) = 0;

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
