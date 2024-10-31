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

namespace barney {
  namespace render {
    struct DeviceMaterial;
    struct HostMaterial;
    
// #define DEFAULT_RADIANCE_FROM_ENV .8f

    // struct QuadLight {
    //   vec3f corner, edge0, edge1, emission;
    //   /*! normal of this lights source; this could obviously be derived
    //     from cross(edge0,edge1), but is handle to have in a
    //     renderer */
    //   vec3f normal;
    //   /*! area of this lights source; this could obviously be derived
    //     from cross(edge0,edge1), but is handle to have in a
    //     renderer */
    //   float area;
    // };

    // struct DirLight {
    //   std::string toString();
    //   vec3f direction;
    //   vec3f radiance;
    // };

  
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
        
        const DeviceMaterial *materials;
        const Sampler::DD    *samplers;
        
        EnvMapLight::DD   envMapLight;
        // Globals::DD     globals;
      };
      EnvMapLight::SP envMapLight;

      World(DevGroup::SP devGroup);
      virtual ~World();

      void set(const std::vector<QuadLight::DD> &quadLights)
      {
        if (quadLights.empty()) 
          owlBufferResize(quadLightsBuffer,1);
        else {
          owlBufferResize(quadLightsBuffer,quadLights.size());
          owlBufferUpload(quadLightsBuffer,quadLights.data());
        }
        numQuadLights = quadLights.size();
      }
      void set(const std::vector<DirLight::DD> &dirLights)
      {
        if (dirLights.empty()) 
          owlBufferResize(dirLightsBuffer,1);
        else {
          owlBufferResize(dirLightsBuffer,dirLights.size());
          owlBufferUpload(dirLightsBuffer,dirLights.data());
        }
        numDirLights = dirLights.size();
      }

      void set(EnvMapLight::SP envMapLight) {
        this->envMapLight = envMapLight;
      }
      
      DD getDD(const Device::SP &device) const;

      // Globals globals;
      MaterialRegistry::SP materialRegistry;
      SamplerRegistry::SP  samplerRegistry;
      OWLBuffer quadLightsBuffer = 0;
      int numQuadLights = 0;
      OWLBuffer dirLightsBuffer = 0;
      int numDirLights = 0;
      DevGroup::SP devGroup;
  private:
      std::shared_ptr<render::HostMaterial> defaultMaterial = 0;
    };

  }
}
