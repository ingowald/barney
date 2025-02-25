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

#include "rtcore/common/rtcore-common.h"
// #include "rtcore/common/Backend.h"
#include "embree4/rtcore.h"

namespace rtc {
  namespace embree {

    using namespace owl::common;
    
    struct LaunchSystem;
    LaunchSystem *createLaunchSystem();

    struct Device;
    struct Denoiser;
    struct Group;
    struct Buffer;
    struct Geom;
    struct GeomType;
    struct TextureData;
    struct Texture;
    
    struct Device {
      Device(int physicalGPU);
      virtual ~Device();

      Denoiser *createDenoiser();

      void destroy();
 
      // /*! returns a string that describes what kind of compute device
      //     this is (eg, "cuda" vs "cpu" */
      // std::string computeType() const { return "cpu"; }
      
      // /*! returns a string that describes what kind of compute device
      //     this is (eg, "optix" vs "embree" */
      // std::string traceType() const { return "embree"; }
     
      // ==================================================================
      // basic compute stuff
      // ==================================================================
      void copyAsync(void *dst, const void *src, size_t numBytes)
      { memcpy(dst,src,numBytes); }
      
      void *allocHost(size_t numBytes)
      { return malloc(numBytes); }
      
      void freeHost(void *mem)
      { free(mem); }
      
      void memsetAsync(void *mem,int value, size_t size)
      { memset(mem,value,size); }
      
      void *allocMem(size_t numBytes)
      { return malloc(numBytes); }
      
      void freeMem(void *mem)
      { free(mem); }
      
      void sync()
      {/*no-op*/}
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const
      {/*no-op*/ return 0; }
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const
      {/*no-op*/}

      TextureData *createTextureData(vec3i dims,
                                     rtc::DataType format,
                                     const void *texels);
      void freeTextureData(TextureData *td);
      void freeTexture(Texture *tex);
      
      // ==================================================================
      // kernels
      // ==================================================================
      // Compute *createCompute(const std::string &name);
      
      // Trace *createTrace(const std::string &name,
      //                         size_t rayGenSize);

      // ==================================================================
      // buffer stuff
      // ==================================================================
      Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0);
     
      void freeBuffer(Buffer *buffer);
      
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
      
      GeomType *createUserGeomType(const char *ptxName,
                                        const char *typeName,
                                        size_t sizeOfDD,
                                        bool has_ah,
                                        bool has_ch);
      
      GeomType *createTrianglesGeomType(const char *ptxName,
                                             const char *typeName,
                                             size_t sizeOfDD,
                                             bool has_ah,
                                             bool has_ch);
      
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
      
      void freeGroup(Group *group);

      LaunchSystem *ls = 0;
      RTCDevice embreeDevice = 0;
    };
    
  }
}
