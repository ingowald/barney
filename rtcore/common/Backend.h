DEPRECATED

// #pragma once

// #include "rtcore/common/rtcore-common.h"

// namespace rtc {
//   // namespace device {
//   //   typedef struct _OpaqueAccel   *AccelHandle;
//   //   typedef struct _OpaqueTextureObject *TextureObject;
//   // };
    
// #if 0
//   typedef enum {
//     UCHAR,
//     USHORT,

//     INT,
//     INT2,
//     INT3,
//     INT4,
      
//     FLOAT,
//     FLOAT2,
//     FLOAT3,
//     FLOAT4,
      
//     UCHAR4,
//     NUM_DATA_TYPES
//   } DataType;

//   // struct Backend;
//   struct Device;
//   struct Geom;
//   struct GeomType;
//   struct Group;
//   struct Device;
    
//   // const void *getSymbol(const std::string &symName);

//   struct Object {
//     Object(Device *device) : device(device) {}
//     virtual ~Object() = default;
//     Device *const device;
//   };
    
//   struct GeomType : public Object {
//     GeomType(Device *device) : Object(device) {}
//     virtual ~GeomType() = default;
      
//     virtual rtc::Geom *createGeom() = 0;
//   };
    
//   struct Group : public Object {
//     Group(Device *device) : Object(device) {}
//     virtual ~Group() = default;
      
//     virtual rtc::device::AccelHandle getDD() const = 0;
      
//     virtual void buildAccel() = 0;
//     virtual void refitAccel() = 0;
//   };

//   struct Buffer : public Object {
//     Buffer(Device *device) : Object(device) {}
//     virtual ~Buffer() = default;
//     virtual void *getDD() const = 0;

//     void upload(const void *hostPtr,
//                 size_t numBytes,
//                 size_t ofs = 0);
//     void uploadAsync(const void *hostPtr,
//                      size_t numBytes,
//                      size_t ofs = 0);
//   };

//   struct Geom : public Object {
//     Geom(Device *device) : Object(device) {}
//     virtual ~Geom() = default;
//     /*! only for user geoms */
//     virtual void setPrimCount(int primCount) = 0;
//     /*! can only get called on triangle type geoms */
//     virtual void setVertices(rtc::Buffer *vertices, int numVertices) = 0;
//     virtual void setIndices(rtc::Buffer *indices, int numIndices) = 0;
//     virtual void setDD(const void *dd) = 0;
//   };

//   struct Texture;
//   struct TextureDesc;
    
//   struct TextureData : public Object {
//     TextureData(Device *device,
//                 const vec3i dims,
//                 const rtc::DataType format)
//       : Object(device), dims(dims), format(format)
//     {
//       assert(format < NUM_DATA_TYPES);
//     }
//     virtual ~TextureData() {}

//     virtual rtc::Texture *
//     createTexture(const rtc::TextureDesc &desc) = 0;
      
//     const vec3i dims;
//     const DataType format;
//   };

//   struct TextureDesc;
    
//   struct Texture : public Object {
//     typedef enum {
//       WRAP,CLAMP,BORDER,MIRROR,
//     } AddressMode;
      
//     typedef enum {
//       FILTER_MODE_POINT,FILTER_MODE_LINEAR,
//     } FilterMode;
    
//     typedef enum {
//       COLOR_SPACE_LINEAR, COLOR_SPACE_SRGB,
//     } ColorSpace;

//     Texture(TextureData *const data,
//             const TextureDesc &desc)
//       : Object(data->device), data(data)
//     {}
//     virtual ~Texture() {}
      
//     const vec3i &getDims() const {
//       return data->dims;
//     }
//     virtual rtc::device::TextureObject getDD() const = 0;

//     TextureData *const data;
//   };

//   struct TextureDesc {
//     rtc::Texture::FilterMode filterMode
//     = Texture::FILTER_MODE_LINEAR;
//     rtc::Texture::AddressMode addressMode[3] = {
//       rtc::Texture::CLAMP,
//       rtc::Texture::CLAMP,
//       rtc::Texture::CLAMP,
//     };
//     const vec4f borderColor = {0.f,0.f,0.f,0.f};
//     bool normalizedCoords = true;
//     rtc::Texture::ColorSpace colorSpace
//     = rtc::Texture::COLOR_SPACE_LINEAR;
//   };

    
//   struct Compute : public Object {
//     Compute(rtc::Device *device) : Object(device) {}
//     virtual void launch(int numBlocks,
//                         int blockSize,
//                         const void *dd) = 0;
//     virtual void launch(vec2i numBlocks,
//                         vec2i blockSize,
//                         const void *dd) = 0;
//     virtual void launch(vec3i numBlocks,
//                         vec3i blockSize,
//                         const void *dd) = 0;
//   };

//   struct Trace : public Object {
//     Trace(rtc::Device *device) : Object(device) {}
//     virtual void launch(vec2i launchDims,
//                         const void *dd) = 0;
//     virtual void launch(int launchDims,
//                         const void *dd) = 0;
//     virtual void sync() = 0;
//   };

//   struct Denoiser {
//     virtual ~Denoiser() = default;
//     /*! if this returns true, the denoiser::run() has to be fed with
//       float4 color and normal data; otherwise, it needs to be fed
//       with float3 color and normal data */
//     virtual void resize(vec2i dims) = 0;
//     virtual void run(vec4f *out_rgba,
//                      vec4f *in_rgba,
//                      vec3f *in_normal,
//                      float blendFactor=0.f) = 0;
//   };
    
//   struct Device {
//     Device(const int physicalID)
//       : physicalID(physicalID)
//     {}
      
//     virtual void destroy() = 0;
      
//     /*! returns a string that describes what kind of compute device
//       this is (eg, "cuda" vs "cpu" */
//     virtual std::string computeType() const = 0;
      
//     /*! returns a string that describes what kind of compute device
//       this is (eg, "optix" vs "embree" */
//     virtual std::string traceType() const = 0;

//     virtual Denoiser *createDenoiser() { return nullptr; }
//     // ==================================================================
//     // control flow related stuff
//     // ==================================================================
      
//     /*! sets this gpu as active, and returns physical ID of GPU that
//       was active before */
//     virtual int setActive() const = 0;
      
//     /*! restores the gpu whose ID was previously returend by setActive() */
//     virtual void restoreActive(int oldActive) const = 0;
      
//     // virtual void launchTrace(const void *ddPtr) = 0;
//     virtual void sync() = 0;
      
//     // ==================================================================
//     // pure compute related stuff
//     // ==================================================================
//     virtual void *allocMem(size_t numBytes) = 0;
//     virtual void freeMem(void *mem) = 0;
//     virtual void *allocHost(size_t numBytes) = 0;
//     virtual void freeHost(void *mem) = 0;
//     virtual void memsetAsync(void *mem,int value, size_t size) = 0;
//     virtual void copyAsync(void *dst, const void *src, size_t size) = 0;
//     void copy(void *dst, const void *src, size_t size)
//     { copyAsync(dst,src,size); sync(); }
      
//     // ==================================================================
//     // kernels
//     // ==================================================================
//     virtual rtc::Compute *createCompute(const std::string &) = 0;
      
//     virtual rtc::Trace *createTrace(const std::string &, size_t) = 0;
      
//     // ==================================================================
//     // buffer stuff
//     // ==================================================================
//     virtual rtc::Buffer *createBuffer(size_t numBytes,
//                                       const void *initValues = 0) = 0;
//     virtual void freeBuffer(rtc::Buffer *) = 0;
      
//     // ==================================================================
//     // texture stuff
//     // ==================================================================

//     virtual rtc::TextureData *
//     createTextureData(vec3i dims,
//                       rtc::DataType format,
//                       const void *texels) = 0;
                        
//     virtual void freeTextureData(rtc::TextureData *) = 0;
//     virtual void freeTexture(rtc::Texture *) = 0;

//     // ==================================================================
//     // ray tracing pipeline related stuff
//     // ==================================================================


//     // ------------------------------------------------------------------
//     // rt pipeline/sbtstuff
//     // ------------------------------------------------------------------
//     virtual void buildPipeline() = 0;
//     virtual void buildSBT() = 0;
      
//     // ------------------------------------------------------------------
//     // geomtype stuff
//     // ------------------------------------------------------------------
//     virtual rtc::GeomType *
//     createUserGeomType(const char *ptxName,
//                        const char *typeName,
//                        size_t sizeOfDD,
//                        bool has_ah,
//                        bool has_ch) = 0;
      
      
//     virtual rtc::GeomType *
//     createTrianglesGeomType(const char *ptxName,
//                             const char *typeName,
//                             size_t sizeOfDD,
//                             bool has_ah,
//                             bool has_ch) = 0;
      
//     virtual void freeGeomType(rtc::GeomType *) = 0;

//     // ------------------------------------------------------------------
//     // geom stuff
//     // ------------------------------------------------------------------
//     virtual void freeGeom(rtc::Geom *) = 0;

//     // ------------------------------------------------------------------
//     // group/accel stuff
//     // ------------------------------------------------------------------
//     virtual rtc::Group *
//     createTrianglesGroup(const std::vector<rtc::Geom *> &geoms) = 0;
      
//     virtual rtc::Group *
//     createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) = 0;

//     virtual rtc::Group *
//     createInstanceGroup(const std::vector<rtc::Group *> &groups,
//                         const std::vector<affine3f> &xfms) = 0;
      
//     virtual void freeGroup(rtc::Group *) = 0;
      
//     const int physicalID;
//   };
    
//   // struct Backend {
//   //   typedef std::shared_ptr<Backend> SP;
//   //   virtual ~Backend() = default;

//   //   // virtual std::vector<rtc::Device *>
//   //   // createDevices(const std::vector<int> &gpuIDs) = 0;

//   //   virtual rtc::Device *createDevice(int gpuID) = 0;
      
//   //   /*! number of 'physical' devices - num cuda capable gpus if cuda
//   //     or optix, or 1 if embree */
//   //   int numPhysicalDevices = 0;
      
//   //   static int getDeviceCount();
//   //   static rtc::Backend *get();
//   // private:
//   //   static rtc::Backend *create();
//   //   static rtc::Backend *singleton;
//   // };

//   inline void Buffer::uploadAsync(const void *hostPtr,
//                                   size_t numBytes,
//                                   size_t ofs)
//   {
//     device->copyAsync(((uint8_t*)getDD())+ofs,hostPtr,numBytes);
//   }
    
//   // extern "C" {
//   //   Backend *createBackend_cuda();
//   //   Backend *createBackend_optix();
//   //   Backend *createBackend_embree();
//   // }
// #endif
// }
