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

#include "rtcore/cudaCommon/Device.h"

namespace rtc {
  namespace cuda {

    using cuda_common::Texture;
    using cuda_common::TextureData;
    using cuda_common::SetActiveGPU;

    struct Buffer;
    struct Device;
    struct Geom;
    struct GeomType;
    struct Group;
    struct Denoiser;
    
    struct Device : public cuda_common::Device{
      Device(int physicalGPU)
        : cuda_common::Device(physicalGPU)
      {}
      virtual ~Device();

      std::string toString() const
      { return "rtc::cuda::Device(physical="+std::to_string(physicalID)+")"; }
      
      void destroy();

      void freeGeomType(GeomType *);

      void freeGeom(Geom *);
      
      Group *
      createTrianglesGroup(const std::vector<Geom *> &geoms);
      
      Group *
      createUserGeomsGroup(const std::vector<Geom *> &geoms);

      Group *
      createInstanceGroup(const std::vector<Group *> &groups,
                          const std::vector<int>      &instIDs,
                          const std::vector<affine3f> &xfms);

      void freeGroup(Group *);
      void buildPipeline();
      void buildSBT();
      Buffer *createBuffer(size_t numBytes,
                           const void *initValues = 0);
      void freeBuffer(Buffer *);
      Denoiser *createDenoiser();
      
    };

    struct Denoiser
    {
      Denoiser(Device* device) : device(device) {}
      ~Denoiser() = default;
      void resize(vec2i dims) {}
      void run(float blendFactor) {}
      Device* const device;

      vec4f *in_rgba = 0;
      vec4f *out_rgba = 0;
      vec3f *in_normal = 0;
    };
    
  }
}

#define RTC_IMPORT_TRACE2D(fileNameBase,name,sizeOfLP)          \
  void rtc_cuda_launch_##name(rtc::Device *device,              \
                              vec2i dims,                       \
                              const void *lpData);              \
                                                                \
  ::rtc::TraceKernel2D *createTrace_##name(rtc::Device *device) \
  {                                                             \
    return new ::rtc::cuda::TraceKernel2D                       \
      (device,sizeOfLP,rtc_cuda_launch_##name);                 \
  }                                                             \
    


