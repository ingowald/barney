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

#include "rtcore/common/Backend.h"
#include "embree4/rtcore.h"

namespace barney {
  namespace embree {

    struct LaunchSystem;
    LaunchSystem *createLaunchSystem();
    
    struct Device : public rtc::Device {
      Device(int physicalGPU);
      virtual ~Device();

      void destroy() override;

      // ==================================================================
      // basic compute stuff
      // ==================================================================
      void copyAsync(void *dst, const void *src, size_t numBytes) override
      { memcpy(dst,src,numBytes); }
      
      void *allocHost(size_t numBytes) override
      { return malloc(numBytes); }
      
      void freeHost(void *mem) override
      { free(mem); }
      
      void memsetAsync(void *mem,int value, size_t size) override
      { memset(mem,value,size); }
      
      void *alloc(size_t numBytes) override
      { return malloc(numBytes); }
      
      void freeMem(void *mem) override
      { free(mem); }
      
      void sync() override
      {/*no-op*/}
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const override
      {/*no-op*/ return 0; }
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const override
      {/*no-op*/}

      rtc::TextureData *createTextureData(vec3i dims,
                                          rtc::DataType format,
                                          const void *texels) override;
      void freeTextureData(rtc::TextureData *td) override;
      void freeTexture(rtc::Texture *tex) override;
      
      // ==================================================================
      // kernels
      // ==================================================================
      rtc::Compute *createCompute(const std::string &name) override;
      
      rtc::Trace *createTrace(const std::string &name,
                              size_t rayGenSize) override;

      // ==================================================================
      // buffer stuff
      // ==================================================================
      rtc::Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0) override;
      
      void freeBuffer(rtc::Buffer *buffer) override;
      
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
      createUserGeomType(const char *typeName,
                         size_t sizeOfDD,
                         const char *boundsFctName,
                         const char *isecFctName,
                         const char *ahFctName,
                         const char *chFctName) 
        override 
      { BARNEY_NYI(); }
      
      rtc::GeomType *createTrianglesGeomType(const char *typeName,
                                             size_t sizeOfDD,
                                             bool has_ah,
                                             bool has_ch) override;
      
      void freeGeomType(rtc::GeomType *) override;

      // ------------------------------------------------------------------
      // geom stuff
      // ------------------------------------------------------------------
      
      void freeGeom(rtc::Geom *) override 
      { BARNEY_NYI(); }
      
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
      
      void freeGroup(rtc::Group *group) override;

      LaunchSystem *ls = 0;
      RTCDevice embreeDevice = 0;
    };
    
  }
}
