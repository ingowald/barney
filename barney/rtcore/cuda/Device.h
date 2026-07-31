// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cudaCommon/Device.h"

namespace BARNEY_NS {
  namespace rtc {

    struct Buffer;
    struct Device;
    struct Geom;
    struct GeomType;
    struct Group;
    struct Denoiser;

    struct TraceKernel2D;

    struct Device : public CudaDeviceBase {
      Device(int physicalGPU)
        : CudaDeviceBase(physicalGPU)
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
      void resize(vec2i dims) { outputDims = dims; }
      void run(float blendFactor) {}
      Device* const device;

      vec4f *in_rgba = 0;
      vec4f *out_rgba = 0;
      vec3f *in_normal = 0;

      bool upscaleMode = false;
      vec2i outputDims = {0,0};
    };

    rtc::AccelHandle getAccelHandle(Group *ig);
  }
}



