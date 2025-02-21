// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "rtcore/cuda/CUDACommon.h"
#include <owl/owl.h>

namespace rtc {
  namespace optix {
    
    using rtc::cuda_common::SetActiveGPU;

    struct Device;
    struct Denoiser;
    struct Group;
    struct Buffer;
    struct Geom;
    struct GeomType;

    using rtc::cuda_common::TextureData;
    using rtc::cuda_common::Texture;
    
    struct TraceKernel1D {
      TraceKernel1D(Device *device,
                    const std::string &ptxCode,
                    const std::string &kernelName);
      void launch(int launchDims,
                  const void *kernelData);
    };
    
    struct Device : public cuda_common::Device {
      Device(int physicalGPU);
      virtual ~Device();
      
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
      createInstanceGroup(const std::vector<Group *> &groups,
                          const std::vector<affine3f> &xfms);

      void freeGroup(Group *);
         
      OWLContext      owl = 0;

      void sync();
      
      std::vector<cudaStream_t> activeTraceStreams;
    };

  }
}


#define RTC_IMPORT_USER_GEOM_TYPE(Type,Class,has_ah,has_ch)     \
  extern "C" char Type##_ptx[];                                 \
  rtc::GeomType *createGeomType_##Type(rtc::Device *device)     \
  {                                                             \
    return new rtc::optix::UserGeomType(device,                 \
                                        Type##_ptx,             \
                                        #Type,                  \
                                        sizeof(Class),          \
                                        has_ah,has_ch);         \
  }

#define RTC_IMPORT_TRIANGLES_GEOM_TYPE(Type,Class,has_ah,has_ch)        \
  extern "C" char Type##_ptx[];                                         \
  rtc::GeomType *createGeomType_##Type(rtc::Device *device)             \
  {                                                                     \
    return new rtc::optix::TrianglesGeomType(device,                    \
                                             Type##_ptx,                \
                                             #Type,                     \
                                             sizeof(Class),             \
                                             has_ah,has_ch);            \
  }


#define RTC_IMPORT_TRACE1D(kernelName,fileNameBase)                   \
  extern "C" char kernelName##_ptx[];                                 \
  rtc::TraceKernel1D *createTrace_##kernelName(rtc::Device *device)     \
  {                                                                     \
    return new rtc::TraceKernel2D(device,kernelName##_ptx,#kernelName); \
  }
# define RTC_EXPORT_TRACE1D(name,RayGenType)                     \
  extern "C"  __global__                                         \
  void __raygen__##name()                                        \
  {                                                              \
    RayGenType *rg = (RayGenType*)optixGetSbtDataPointer();      \
    ::rtc::optix::TraceInterface rtcore;                      \
    rg->run(rtcore);                                             \
  }                                                                   
    


