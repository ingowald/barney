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

#include "rtcore/cuda/CUDABackend.h"

namespace barney {
  namespace optix {
    
    struct OptixBackend;
    using barney::cuda::SetActiveGPU;
    
    struct Device : public cuda::BaseDevice {
      Device(int physicalGPU);
      virtual ~Device();
      
      void destroy() override;

      /*! returns a string that describes what kind of compute device
          this is (eg, "cuda" vs "cpu" */
      std::string computeType() const override { return "cuda"; }
      
      /*! returns a string that describes what kind of compute device
          this is (eg, "optix" vs "embree" */
      std::string traceType() const override { return "optix"; }
      
      // ==================================================================
      // denoiser
      // ==================================================================
      rtc::Denoiser *createDenoiser() override;

      // ==================================================================
      // kernels
      // ==================================================================
      rtc::Compute *
      createCompute(const std::string &) override;
      
      rtc::Trace *
      createTrace(const std::string &, size_t) override;

      // ==================================================================
      // buffer stuff
      // ==================================================================
      rtc::Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0) override;
      void freeBuffer(rtc::Buffer *) override;
      
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

      void buildPipeline() override;
      void buildSBT() override;

      // ------------------------------------------------------------------
      // geomtype stuff
      // ------------------------------------------------------------------
      
      rtc::GeomType *
      createUserGeomType(const char *ptxName,
                         const char *typeName,
                         size_t sizeOfDD,
                         bool has_ah,
                         bool has_ch) 
        override;
      
      rtc::GeomType *
      createTrianglesGeomType(const char *ptxName,
                              const char *typeName,
                              size_t sizeOfDD,
                              bool has_ah,
                              bool has_ch) override;
      
      void freeGeomType(rtc::GeomType *) override 
      { BARNEY_NYI(); };

      // ------------------------------------------------------------------
      // geom stuff
      // ------------------------------------------------------------------
      
      void freeGeom(rtc::Geom *) override 
      {
        std::cout << "FREEGEOM - MISSING!" << std::endl;
        BARNEY_NYI();
      };
      
      // ------------------------------------------------------------------
      // group/accel stuff
      // ------------------------------------------------------------------
      rtc::Group *
      createTrianglesGroup(const std::vector<rtc::Geom *> &geoms) override;
      
      rtc::Group *
      createUserGeomsGroup(const std::vector<rtc::Geom *> &geoms) override;

      rtc::Group *
      createInstanceGroup(const std::vector<rtc::Group *> &groups,
                          const std::vector<affine3f> &xfms) override;

      void freeGroup(rtc::Group *) override;
         
      OWLContext      owl = 0;

      void sync() override;
      
      std::vector<cudaStream_t> activeTraceStreams;
    };
    
  }
}
