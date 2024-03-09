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
#include "barney/material/Globals.h"

namespace barney {
  namespace render {
    
    struct QuadLight {
      vec3f corner, edge0, edge1, emission;
      /*! normal of this lights source; this could obviously be derived
        from cross(edge0,edge1), but is handle to have in a
        renderer */
      vec3f normal;
      /*! area of this lights source; this could obviously be derived
        from cross(edge0,edge1), but is handle to have in a
        renderer */
      float area;
    };

    struct DirLight {
      std::string toString();
      vec3f direction;
      vec3f radiance;
    };

    struct EnvMapLight {
      struct DD {
        affine3f   transform;
        cudaTextureObject_t texture;
      };
      affine3f   transform;
      OWLTexture texture;
    };
  
    /*! the rendering/path racing related part of a model that describes
      global render settings like light sources, background, envmap,
      etc */
    struct World {
      struct DD {
        int         numQuadLights = 0;
        QuadLight  *quadLights    = nullptr;
        int         numDirLights  = 0;
        DirLight   *dirLights     = nullptr;
        EnvMapLight::DD envMapLight;
        Globals::DD     globals;
      };
      EnvMapLight envMapLight;

      World(DevGroup *devGroup)
        : devGroup(devGroup),
          globals(devGroup)
      {
        quadLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
                                                 OWL_USER_TYPE(QuadLight),
                                                 1,nullptr);
        dirLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
                                                OWL_USER_TYPE(DirLight),
                                                1,nullptr);
        
      }
      void set(const std::vector<QuadLight> &quadLights)
      {
        if (quadLights.empty()) 
          owlBufferResize(quadLightsBuffer,1);
        else {
          owlBufferResize(quadLightsBuffer,quadLights.size());
          owlBufferUpload(quadLightsBuffer,quadLights.data());
        }
        numQuadLights = quadLights.size();
      }
      void set(const std::vector<DirLight> &dirLights)
      {
        if (dirLights.empty()) 
          owlBufferResize(dirLightsBuffer,1);
        else {
          owlBufferResize(dirLightsBuffer,dirLights.size());
          owlBufferUpload(dirLightsBuffer,dirLights.data());
        }
        numDirLights = dirLights.size();
      }
      void set(EnvMapLight envMapLight) {
        this->envMapLight = envMapLight;
      };
      DD getDD(const Device::SP &device) const {
        DD dd;
        dd.quadLights = (QuadLight *)owlBufferGetPointer(quadLightsBuffer,device->owlID);
        dd.numQuadLights = numQuadLights;
        dd.dirLights = (DirLight *)owlBufferGetPointer(dirLightsBuffer,device->owlID);
        dd.numDirLights = numDirLights;
        if (envMapLight.texture)
          dd.envMapLight.texture 
            = owlTextureGetObject(envMapLight.texture,device->owlID);
        else 
          dd.envMapLight.texture
            = 0;
        dd.envMapLight.transform = envMapLight.transform;
        dd.globals = globals.getDD(device);
        return dd;
      }

      Globals globals;
      OWLBuffer quadLightsBuffer = 0;
      int numQuadLights = 0;
      OWLBuffer dirLightsBuffer = 0;
      int numDirLights = 0;
      DevGroup *const devGroup;
    };

  }
}
