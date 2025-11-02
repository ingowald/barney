// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/DeviceGroup.h"
#include "barney/render/Sampler.h"
#include "barney/light/EnvMap.h"
#include "barney/light/DirLight.h"
#include "barney/light/PointLight.h"
#include "barney/light/QuadLight.h"

namespace BARNEY_NS {
  struct SlotContext;
  
  namespace render {

    struct DeviceMaterial;
    struct HostMaterial;

    /*! the rendering/path racing related part of a model that describes
      global render settings like light sources, background, envmap,
      etc */
    struct World {
      typedef std::shared_ptr<World> SP;
      
      struct DD {
        int                  numQuadLights = 0;
        const QuadLight::DD *quadLights    = nullptr;
        int                  numDirLights  = 0;
        const DirLight::DD  *dirLights     = nullptr;
        int                  numPointLights  = 0;
        const PointLight::DD  *pointLights     = nullptr;
        int                 *instIDToUserInstID = 0;
        
        const DeviceMaterial *materials;
        const Sampler::DD    *samplers;
        const rtc::float4    *instanceAttributes[5];
        EnvMapLight::DD       envMapLight;
        // uint32_t              rngSeed;
        uint32_t              rank;
      };
      struct {
        EnvMapLight::SP light;
        affine3f        xfm;
      } envMapLight;

      World(SlotContext *slotContext);
      virtual ~World();

      void set(const std::vector<QuadLight::DD> &quadLights);
      void set(const std::vector<DirLight::DD> &dirLights);
      void set(const std::vector<PointLight::DD> &pointLights);
      void set(EnvMapLight::SP envMapLight, const affine3f &xfm);
      
      PODData::SP instanceAttributes[5];
      PODData::SP instanceUserIDs;
      
      DD getDD(Device *device// , int rngSeed
               );

      struct PLD {
        QuadLight::DD *quadLights = 0;
        int numQuadLights = 0;
        DirLight::DD *dirLights = 0;
        int numDirLights = 0;
        PointLight::DD *pointLights = 0;
        int numPointLights = 0;
      };
      PLD *getPLD(Device *device);
      
      std::vector<PLD> perLogical;
      DevGroup::SP const devices;
      SlotContext *const slotContext;
    };

  }
}
