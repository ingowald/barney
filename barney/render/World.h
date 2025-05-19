// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/DeviceGroup.h"
// #include "barney/material/Globals.h"
// #include "barney/render/DeviceMaterial.h"
#include "barney/render/Sampler.h"
// #include "barney/material/DeviceMaterial.h"
#include "barney/light/EnvMap.h"
#include "barney/light/DirLight.h"
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
      };
      PLD *getPLD(Device *device);
      
      std::vector<PLD> perLogical;
      DevGroup::SP const devices;
      SlotContext *const slotContext;
    };

  }
}
