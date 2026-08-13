// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cudaCommon/Device.h"
#include <owl/owl.h>

namespace BARNEY_NS {
  namespace rtc {

    struct Device;
    struct Denoiser;
    struct Group;
    struct Buffer;
    struct Geom;
    struct GeomType;

    rtc::AccelHandle getAccelHandle(Group *ig);
    
    struct TraceKernel2D {
      TraceKernel2D(Device *device,
                    const std::string &ptxCode,
                    const std::string &kernelName,
                    size_t sizeOfLP);
      void launch(vec2i launchDims,
                  const void *kernelData);
      Device *const device;
      OWLModule mod;
      OWLRayGen rg;
      OWLParams lp;
      cudaStream_t lpStream;
    };
    
    struct Device : public CudaDeviceBase {
      Device(int physicalGPU);
      ~Device() override;

      std::string toString() const
      { return "optix::Device(physical="+std::to_string(physicalID)+")"; }
      
      void destroy();

      /*! returns a string that describes what kind of compute device
          this is (eg, "cuda" vs "cpu" */
      std::string computeType() const { return "cuda"; }
      
      /*! returns a string that describes what kind of compute device
          this is (eg, "optix" vs "embree" */
      std::string traceType() const { return "optix"; }
      
      // ==================================================================
      // denoiser
      // ==================================================================
      Denoiser *createDenoiser();

      // ==================================================================
      // kernels
      // ==================================================================
      // rtc::Compute *
      // createCompute(const std::string &);
      
      // rtc::Trace *
      // createTrace(const std::string &, size_t);

      // ==================================================================
      // buffer stuff
      // ==================================================================
      Buffer *createBuffer(size_t numBytes,
                           const void *initValues = 0);
      void freeBuffer(Buffer *);
      
      // ==================================================================
      // texture stuff
      // ==================================================================

      // in parent

      // ==================================================================
      // ray tracing pipeline related stuff
      // ==================================================================


      // ------------------------------------------------------------------
      // rt pipeline/sbtstuff
      // ------------------------------------------------------------------

      void buildPipeline();
      void buildSBT();

      // ------------------------------------------------------------------
      // geomtype stuff
      // ------------------------------------------------------------------
      
      // rtc::GeomType *
      // createUserGeomType(const char *ptxName,
      //                    const char *typeName,
      //                    size_t sizeOfDD,
      //                    bool has_ah,
      //                    bool has_ch) 
      //  ;
      
      // rtc::GeomType *
      // createTrianglesGeomType(const char *ptxName,
      //                         const char *typeName,
      //                         size_t sizeOfDD,
      //                         bool has_ah,
      //                         bool has_ch);
      
      void freeGeomType(GeomType *);

      // ------------------------------------------------------------------
      // geom stuff
      // ------------------------------------------------------------------
      
      void freeGeom(Geom *);
      
      // ------------------------------------------------------------------
      // group/accel stuff
      // ------------------------------------------------------------------
      Group *
      createTrianglesGroup(const std::vector<Geom *> &geoms);
      
      Group *
      createUserGeomsGroup(const std::vector<Geom *> &geoms);

      Group *
      createInstanceGroup(const std::vector<Group *>  &groups,
                          const std::vector<int>      &instIDs,
                          const std::vector<affine3f> &xfms);

      void freeGroup(Group *);
         
      OWLContext      owl = 0;

      void sync();

      std::vector<cudaStream_t> activeTraceStreams;
      bool programsDirty = true;
    };

  }
} 


#define RTC_IMPORT_USER_GEOM(moduleName,typeName,DDType,has_ah,has_ch)  \
  extern "C" char moduleName##_ptx[];                                   \
  rtc::GeomType *                                                       \
  createGeomType_##typeName(::BARNEY_NS::rtc::Device *device)           \
  {                                                                     \
    return new ::BARNEY_NS::rtc::UserGeomType(device,                   \
                                              moduleName##_ptx,         \
                                              #typeName,                \
                                              sizeof(DDType),           \
                                              has_ah,has_ch);           \
  }

#define RTC_IMPORT_TRIANGLES_GEOM(moduleName,typeName,DDType,has_ah,has_ch) \
  extern "C" char moduleName##_ptx[];                                   \
  ::BARNEY_NS::rtc::GeomType *                                          \
  createGeomType_##typeName(rtc::Device *device)                        \
  {                                                                     \
    return new ::BARNEY_NS::rtc::TrianglesGeomType                      \
      (device,                                                          \
       moduleName##_ptx,                                                \
       #typeName,                                                       \
       sizeof(DDType),                                                  \
       has_ah,has_ch);                                                  \
  }



# define RTC_EXPORT_TRACE2D(name,RayGenType)                     \
  extern "C"  __global__                                         \
  void __raygen__##name()                                        \
  {                                                              \
    RayGenType *rg = (RayGenType*)optixGetSbtDataPointer();          \
    ::BARNEY_NS::rtc::TraceInterface rtcore;                         \
    rg->run(rtcore);                                                 \
  }

#define RTC_IMPORT_TRACE2D(fileNameBase,kernelName,sizeOfLP)            \
  extern "C" char fileNameBase##_ptx[];                                 \
  ::BARNEY_NS::rtc::TraceKernel2D *                                     \
  createTrace_##kernelName(::BARNEY_NS::rtc::Device *device)            \
  {                                                                     \
    return new ::BARNEY_NS::rtc::TraceKernel2D                          \
      (device,fileNameBase##_ptx,                                       \
       #kernelName,sizeOfLP);                                           \
  }

