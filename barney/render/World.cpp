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
#include "barney/Context.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/material/Material.h"
#include "barney/render/MaterialRegistry.h"
#include "barney/render/SamplerRegistry.h"

namespace barney {
  namespace render {

    World::World(SlotContext *slotContext)
      : devices(slotContext->devices),
        slotContext(slotContext)
    {
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        auto rtc = device->rtc;
        pld->quadLightsBuffer
          = rtc->createBuffer(sizeof(QuadLight::DD));
        pld->dirLightsBuffer
          = rtc->createBuffer(sizeof(DirLight::DD));
      }
    }
    
    World::~World()
    {}

    World::DD World::getDD(Device *device) 
    {
      PLD *pld = getPLD(device);
      auto rtc = device->rtc;
      DD dd;
      dd.quadLights
        = (QuadLight::DD *)pld->quadLightsBuffer->getDD();
      dd.numQuadLights = pld->numQuadLights;
      dd.dirLights
        = (DirLight::DD *)pld->dirLightsBuffer->getDD();
      dd.numDirLights = pld->numDirLights;
      
      dd.envMapLight
        = envMapLight.light
        ? envMapLight.light->getDD(device,envMapLight.xfm)
        : EnvMapLight::DD{};
      
      dd.samplers  = slotContext->samplerRegistry->getDD(device);
      dd.materials = slotContext->materialRegistry->getDD(device);

      return dd;
    }

    MaterialRegistry::MaterialRegistry(const DevGroup::SP &devices)
      : devices(devices)
    {
      numReserved = 1;
      perLogical.resize(devices->numLogical);
      
      for (auto device : *devices)
        getPLD(device)->buffer
          = device->rtc->createBuffer(numReserved*sizeof(DeviceMaterial));
    }

    MaterialRegistry::~MaterialRegistry()
    {
      // owlBufferRelease(buffer);
      for (auto device : *devices)
        device->rtc->free(getPLD(device)->buffer);
    }
    
    void MaterialRegistry::grow()
    {
      size_t oldNumBytes = numReserved * sizeof(DeviceMaterial);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(DeviceMaterial);
      for (auto device : *devices) {
      // ------------------------------------------------------------------
      // save old materials
      // ------------------------------------------------------------------
        PLD *pld = getPLD(device);
        rtc::Buffer *oldBuffer = pld->buffer;
        auto rtc = device->rtc;
        
        rtc::Buffer *newBuffer
          = rtc->createBuffer(newNumBytes);
        rtc->copyAsync(newBuffer->getDD(),oldBuffer->getDD(),oldNumBytes);
        rtc->sync();
        rtc->free(oldBuffer);
        pld->buffer = newBuffer;
      }
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
  
    void MaterialRegistry::setMaterial(int materialID,
                                       const DeviceMaterial &dd,
                                       Device *device)
    {
      // for (auto device : *devices) {
        PLD *pld = getPLD(device);
        pld->buffer->upload(&dd,sizeof(dd),sizeof(dd)*materialID);
      // }
    }



    SamplerRegistry::SamplerRegistry(const DevGroup::SP &devices)
      : devices(devices)
    {
      numReserved = 1;
      perLogical.resize(devices->numLogical);
      
      for (auto device : *devices)
        getPLD(device)->buffer
          = device->rtc->createBuffer(numReserved*sizeof(Sampler::DD));
    }

    SamplerRegistry::~SamplerRegistry()
    {
      for (auto device : *devices)
        device->rtc->freeBuffer(getPLD(device)->buffer);
    }
    
    void SamplerRegistry::grow()
    {
      size_t oldNumBytes = numReserved * sizeof(Sampler::DD);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(Sampler::DD);
      
      for (auto device : *devices) {
        // ------------------------------------------------------------------
        // save old samplers
        // ------------------------------------------------------------------
        PLD *pld = getPLD(device);
        rtc::Buffer *oldBuffer = pld->buffer;
        auto rtc = device->rtc;
        
        rtc::Buffer *newBuffer
          = rtc->createBuffer(newNumBytes);
        rtc->copyAsync(newBuffer->getDD(),oldBuffer->getDD(),oldNumBytes);
        rtc->sync();
        rtc->free(oldBuffer);
        pld->buffer = newBuffer;
      }
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
  
    // const Sampler::DD *SamplerRegistry::getPointer(int owlDeviceID) const
    // {
    //   // return (Sampler::DD *)owlBufferGetPointer(buffer,owlDeviceID);
    //   return (Sampler::DD *)owlBufferGetPointer(buffer,owlDeviceID);
    // }    

    void SamplerRegistry::setDD(int samplerID,
                                const Sampler::DD &dd,
                                Device *device)
    {
      size_t offset = samplerID*sizeof(dd);
      getPLD(device)->buffer->upload(&dd,sizeof(dd),offset);
    }

    
    void World::set(const std::vector<QuadLight::DD> &quadLights)
    {
      for (auto device : *devices) {
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeBuffer(pld->quadLightsBuffer);
        pld->quadLightsBuffer
          = rtc->createBuffer(quadLights.size()*sizeof(quadLights[0]),
                              quadLights.data());
        pld->numQuadLights = (int)quadLights.size();
      }
    }
    
    void World::set(const std::vector<DirLight::DD> &dirLights)
    {
      for (auto device : *devices) {
        auto pld = getPLD(device);
        auto rtc = device->rtc;
        rtc->freeBuffer(pld->dirLightsBuffer);
        pld->dirLightsBuffer
          = rtc->createBuffer(dirLights.size()*sizeof(dirLights[0]),
                              dirLights.data());
        pld->numDirLights = (int)dirLights.size();
      }
    }

    void World::set(EnvMapLight::SP envMapLight, const affine3f &xfm)
    {
      this->envMapLight.light = envMapLight;
      this->envMapLight.xfm = xfm;
    }
      
    
  }
}
