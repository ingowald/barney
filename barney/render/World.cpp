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
#include "barney/DeviceContext.h"
#include "barney/material/DeviceMaterial.h"
#include "barney/material/Material.h"

namespace barney {
  namespace render {

    World::World(DevGroup::SP devGroup)
      : devGroup(devGroup)
        // globals(devGroup)
    {
      auto rtc = getRTC();
      quadLightsBuffer = rtc->createBuffer(sizeof(QuadLight::DD));
      dirLightsBuffer = rtc->createBuffer(sizeof(DirLight::DD));
      // quadLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
      //                                          OWL_USER_TYPE(QuadLight),
      //                                          1,nullptr);
      // dirLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
      //                                         OWL_USER_TYPE(DirLight),
      //                                         1,nullptr);
    }
    World::~World()
    {}

    World::DD World::getDD(const Device::SP &device) const
    {
      DD dd;
      dd.quadLights
        // = (QuadLight::DD *)owlBufferGetPointer(quadLightsBuffer,device->owlID);
        = (QuadLight::DD *)quadLightsBuffer->getDD(device->rtc);
      dd.numQuadLights = numQuadLights;
      dd.dirLights
        // = (DirLight::DD *)owlBufferGetPointer(dirLightsBuffer,device->owlID);
        = (DirLight::DD *)dirLightsBuffer->getDD(device->rtc);
      dd.numDirLights = numDirLights;

      dd.envMapLight
        = envMapLight
        ? envMapLight->getDD(device)
        : EnvMapLight::DD{};
      
      // dd.samplers  = samplerRegistry->getPointer(device->owlID);
      // dd.materials = materialRegistry->getPointer(device->owlID);

      return dd;
    }

    MaterialRegistry::MaterialRegistry(DevGroup::SP devGroup)
      : devGroup(devGroup)
    {
      auto rtc = getRTC();
      numReserved = 1;
      
      // buffer = owlDeviceBufferCreate
      //   (devGroup->owl,OWL_USER_TYPE(DeviceMaterial),numReserved,nullptr);
      buffer = rtc->createBuffer(numReserved*sizeof(DeviceMaterial));
    }

    MaterialRegistry::~MaterialRegistry()
    {
      // owlBufferRelease(buffer);
      devGroup->rtc->free(buffer);
    }
    
    void MaterialRegistry::grow()
    {
      assert(this->buffer);
      
      // ------------------------------------------------------------------
      // save old materials
      // ------------------------------------------------------------------
      rtc::Buffer *oldBuffer = this->buffer;
      size_t oldNumBytes = numReserved * sizeof(DeviceMaterial);
      auto rtc = getRTC();
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(DeviceMaterial);

      rtc::Buffer *newBuffer
        = rtc->createBuffer(newNumBytes);
      rtc->copy(newBuffer,oldBuffer,oldNumBytes);
      rtc->free(oldBuffer);
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
  
    // const DeviceMaterial *MaterialRegistry::getPointer(int owlDeviceID) const 
    // {
    //   return (DeviceMaterial *)owlBufferGetPointer(buffer,owlDeviceID);
    // }    


    void MaterialRegistry::setMaterial(int materialID,
                                      const DeviceMaterial &dd,
                                      int deviceID)
    {
      buffer->upload(&dd,sizeof(dd),sizeof(dd)*materialID);
      // BARNEY_CUDA_CALL(Memcpy((void*)(getPointer(deviceID)+materialID),
      //                         &dd,sizeof(dd),cudaMemcpyDefault));
    }



    SamplerRegistry::SamplerRegistry(DevGroup::SP devGroup)
      : devGroup(devGroup)
    {
      numReserved = 1;
      // buffer = owlDeviceBufferCreate
      //   (devGroup->owl,OWL_USER_TYPE(Sampler::DD),1,nullptr);
      buffer = getRTC()->createBuffer(sizeof(Sampler::DD));
    }

    SamplerRegistry::~SamplerRegistry()
    {
      // owlBufferRelease(buffer);
      getRTC()->free(buffer);
    }
    
    void SamplerRegistry::grow()
    {
      auto rtc = getRTC();

      size_t oldNumBytes = numReserved * sizeof(Sampler::DD);
      numReserved *= 2;
      size_t newNumBytes = numReserved * sizeof(Sampler::DD);
      rtc::Buffer *newBuffer
        = rtc->createBuffer(newNumBytes);
      rtc->copy(newBuffer,this->buffer,oldNumBytes);
      rtc->free(this->buffer);
      this->buffer = newBuffer;
      // // ------------------------------------------------------------------
      // // save old materials
      // // ------------------------------------------------------------------
      // OWLBuffer tmp = owlDeviceBufferCreate
      //   (devGroup->owl,OWL_USER_TYPE(Sampler::DD),numReserved,nullptr);
      // for (int i=0;i<devGroup->size();i++) {
      //   BARNEY_CUDA_CALL(Memcpy((void*)owlBufferGetPointer(tmp,i),
      //                           (void*)owlBufferGetPointer(buffer,i),
      //                           oldNumBytes,cudaMemcpyDefault));
      // }

      // // ------------------------------------------------------------------
      // // resize backing storage
      // // ------------------------------------------------------------------
      // owlBufferResize(buffer,numReserved);

      // // ------------------------------------------------------------------
      // // and restore old values into resized storage
      // // ------------------------------------------------------------------
      // for (int i=0;i<devGroup->size();i++) {
      //   BARNEY_CUDA_CALL(Memcpy((void*)owlBufferGetPointer(buffer,i),
      //                           (void*)owlBufferGetPointer(tmp,i),
      //                           oldNumBytes,cudaMemcpyDefault));
      // }
      // owlBufferRelease(tmp);
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
                                rtc::Device *device)
    {
      buffer->upload(&dd,sizeof(dd),/*offset*/samplerID*sizeof(dd),device);
      // BARNEY_CUDA_CALL(Memcpy((void*)(getPointer(deviceID)+samplerID),
      //                         &dd,sizeof(dd),cudaMemcpyDefault));
    }


      void World::set(const std::vector<QuadLight::DD> &quadLights)
      {
        barney::rtc::resizeAndUpload(quadLightsBuffer,quadLights);
        // rtccore
        // if (quadLights.empty())
        //   // owlBufferResize(quadLightsBuffer,1);
        //   quadLightsBuffer->resize(sizeof(QuadLight::DD));
        // else {
        //   // owlBufferResize(quadLightsBuffer,quadLights.size());
        //   // owlBufferUpload(quadLightsBuffer,quadLights.data());
        //   quadLightsBuffer->resizeAndUpload(quadLights);
        // }
        numQuadLights = (int)quadLights.size();
      }
    
      void World::set(const std::vector<DirLight::DD> &dirLights)
      {
        barney::rtc::resizeAndUpload(dirLightsBuffer,dirLights);
        // if (dirLights.empty()) 
        //   owlBufferResize(dirLightsBuffer,1);
        // else {
        //   owlBufferResize(dirLightsBuffer,dirLights.size());
        //   owlBufferUpload(dirLightsBuffer,dirLights.data());
        // }
        numDirLights = (int)dirLights.size();
      }

    void World::set(EnvMapLight::SP envMapLight)
    {
        this->envMapLight = envMapLight;
      }
      
    
  }
}
