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

#include "barney/render/World.h"
#include "barney/render/DeviceMaterial.h"

namespace barney {
  namespace render {

    World::World(DevGroup::SP devGroup)
      : devGroup(devGroup),
        materialRegistry(std::make_shared<MaterialRegistry>(devGroup)),
        samplerRegistry(std::make_shared<SamplerRegistry>(devGroup))
        // globals(devGroup)
    {
      quadLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
                                               OWL_USER_TYPE(QuadLight),
                                               1,nullptr);
      dirLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
                                              OWL_USER_TYPE(DirLight),
                                              1,nullptr);
    }
    World::~World()
    {}

    EnvMapLight::DD EnvMapLight::getDD(const Device::SP &device) const
    {
      DD dd;
      dd.dims = dims;
      if (texture) {
        dd.texture 
          = owlTextureGetObject(texture,device->owlID);
      } else 
        dd.texture = 0;
      
      dd.toWorld = toWorld;
      dd.toLocal = toLocal;
      dd.cdf_y = (const float *)owlBufferGetPointer(cdf_y,device->owlID);
      dd.allCDFs_x = (const float *)owlBufferGetPointer(allCDFs_x,device->owlID);
      return dd;
    }

    World::DD World::getDD(const Device::SP &device) const
    {
      DD dd;
      dd.quadLights = (QuadLight *)owlBufferGetPointer(quadLightsBuffer,device->owlID);
      dd.numQuadLights = numQuadLights;
      dd.dirLights = (DirLight *)owlBufferGetPointer(dirLightsBuffer,device->owlID);
      dd.numDirLights = numDirLights;

      dd.envMapLight = envMapLight.getDD(device);
      // dd.globals = globals.getDD(device);
      dd.radiance  = radiance;
      dd.samplers  = samplerRegistry->getPointer(device->owlID);
      dd.materials = materialRegistry->getPointer(device->owlID);

      return dd;
    }

    MaterialRegistry::MaterialRegistry(DevGroup::SP devGroup)
      : devGroup(devGroup)
    {
      numReserved = 1;
      buffer = owlDeviceBufferCreate
        (devGroup->owl,OWL_USER_TYPE(DeviceMaterial),numReserved,nullptr);
    }

    MaterialRegistry::~MaterialRegistry()
    {
      owlBufferRelease(buffer);
    }
    
    void MaterialRegistry::grow()
    {
      // ------------------------------------------------------------------
      // save old materials
      // ------------------------------------------------------------------
      size_t oldNumBytes = numReserved * sizeof(DeviceMaterial);
      OWLBuffer tmp = owlDeviceBufferCreate
        (devGroup->owl,OWL_USER_TYPE(DeviceMaterial),numReserved,nullptr);
      for (int i=0;i<devGroup->size();i++) {
        BARNEY_CUDA_CALL(Memcpy((void*)owlBufferGetPointer(tmp,i),
                                (void*)owlBufferGetPointer(buffer,i),
                                oldNumBytes,cudaMemcpyDefault));
      }

      // ------------------------------------------------------------------
      // resize backing storage
      // ------------------------------------------------------------------
      numReserved *= 2;
      owlBufferResize(buffer,numReserved);

      // ------------------------------------------------------------------
      // and restore old values into resized storage
      // ------------------------------------------------------------------
      for (int i=0;i<devGroup->size();i++) {
        BARNEY_CUDA_CALL(Memcpy((void*)owlBufferGetPointer(buffer,i),
                                (void*)owlBufferGetPointer(tmp,i),
                                oldNumBytes,cudaMemcpyDefault));
      }
      owlBufferRelease(tmp);
    }

    int MaterialRegistry::allocate()
    {
      if (!reusableIDs.empty()) {
        int ID = reusableIDs.top();
        reusableIDs.pop();
        return ID;
      }
      if (nextFree == numReserved) grow();

      return nextFree++;
    }
   
    void MaterialRegistry::release(int nowReusableID)
    {
      reusableIDs.push(nowReusableID);
    }
  
    const DeviceMaterial *MaterialRegistry::getPointer(int owlDeviceID) const 
    {
      return (DeviceMaterial *)owlBufferGetPointer(buffer,owlDeviceID);
    }    


    void MaterialRegistry::setMaterial(int materialID,
                                      const DeviceMaterial &dd,
                                      int deviceID)
    {
      BARNEY_CUDA_CALL(Memcpy((void*)(getPointer(deviceID)+materialID),
                              &dd,sizeof(dd),cudaMemcpyDefault));
    }



    SamplerRegistry::SamplerRegistry(DevGroup::SP devGroup)
      : devGroup(devGroup)
    {
      numReserved = 1;
      buffer = owlDeviceBufferCreate
        (devGroup->owl,OWL_USER_TYPE(Sampler::DD),1,nullptr);
    }

    SamplerRegistry::~SamplerRegistry()
    {
      owlBufferRelease(buffer);
    }
    
    void SamplerRegistry::grow()
    {
      // ------------------------------------------------------------------
      // save old materials
      // ------------------------------------------------------------------
      size_t oldNumBytes = numReserved * sizeof(Sampler::DD);
      OWLBuffer tmp = owlDeviceBufferCreate
        (devGroup->owl,OWL_USER_TYPE(Sampler::DD),numReserved,nullptr);
      for (int i=0;i<devGroup->size();i++) {
        BARNEY_CUDA_CALL(Memcpy((void*)owlBufferGetPointer(tmp,i),
                                (void*)owlBufferGetPointer(buffer,i),
                                oldNumBytes,cudaMemcpyDefault));
      }

      // ------------------------------------------------------------------
      // resize backing storage
      // ------------------------------------------------------------------
      numReserved *= 2;
      owlBufferResize(buffer,numReserved);

      // ------------------------------------------------------------------
      // and restore old values into resized storage
      // ------------------------------------------------------------------
      for (int i=0;i<devGroup->size();i++) {
        BARNEY_CUDA_CALL(Memcpy((void*)owlBufferGetPointer(buffer,i),
                                (void*)owlBufferGetPointer(tmp,i),
                                oldNumBytes,cudaMemcpyDefault));
      }
      owlBufferRelease(tmp);
    }

    int SamplerRegistry::allocate()
    {
      if (!reusableIDs.empty()) {
        int ID = reusableIDs.top();
        reusableIDs.pop();
        return ID;
      }
      if (nextFree == numReserved) grow();

      return nextFree++;
    }
   
    void SamplerRegistry::release(int nowReusableID)
    {
      reusableIDs.push(nowReusableID);
    }
  
    const Sampler::DD *SamplerRegistry::getPointer(int owlDeviceID) const
    {
      return (Sampler::DD *)owlBufferGetPointer(buffer,owlDeviceID);
    }    

    void SamplerRegistry::setDD(int samplerID,
                               const Sampler::DD &dd,
                               int deviceID)
    {
      BARNEY_CUDA_CALL(Memcpy((void*)(getPointer(deviceID)+samplerID),
                              &dd,sizeof(dd),cudaMemcpyDefault));
    }
    
  }
}
