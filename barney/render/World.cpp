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
        materialLibrary(std::make_shared<MaterialLibrary>(devGroup)),
        samplerLibrary(std::make_shared<SamplerLibrary>(devGroup))
        // globals(devGroup)
    {
      std::cout << "WORLD " << this << " CREATED" << std::endl;
      PING;
      quadLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
                                               OWL_USER_TYPE(QuadLight),
                                               1,nullptr);
      dirLightsBuffer = owlDeviceBufferCreate(devGroup->owl,
                                              OWL_USER_TYPE(DirLight),
                                              1,nullptr);
      PING;
    }
    World::~World()
    {
      std::cout << "WORLD " << this << " IS DYING" << std::endl;
    }
    
    MaterialLibrary::MaterialLibrary(DevGroup::SP devGroup)
      : devGroup(devGroup)
    {
      PING;
      numReserved = 1;
      PRINT(devGroup);
      PRINT(devGroup->owl);
      buffer = owlDeviceBufferCreate
        (devGroup->owl,OWL_USER_TYPE(DeviceMaterial),numReserved,nullptr);
      PING;
      std::cout << "WORLD matlib " << this << " CREATED" << std::endl;
    }

    MaterialLibrary::~MaterialLibrary()
    {
      std::cout << "WORLD matlib " << this << " IS DYING" << std::endl;
      PING;
      owlBufferRelease(buffer);
    }
    
    void MaterialLibrary::grow()
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

    int MaterialLibrary::allocate()
    {
      if (!reusableIDs.empty()) {
        int ID = reusableIDs.top();
        reusableIDs.pop();
        return ID;
      }
      if (nextFree == numReserved) grow();

      return nextFree++;
    }
   
    void MaterialLibrary::release(int nowReusableID)
    {
      std::cout << "WORLD matlib " << this << " release mat!?" << std::endl;
      reusableIDs.push(nowReusableID);
    }
  
    const DeviceMaterial *MaterialLibrary::getPointer(int owlDeviceID) const 
    {
      return (DeviceMaterial *)owlBufferGetPointer(buffer,owlDeviceID);
    }    


    void MaterialLibrary::setMaterial(int materialID,
                                      const DeviceMaterial &dd,
                                      int deviceID)
    {
      PING;
      PRINT(materialID);
      PRINT(numReserved);
      PRINT(deviceID);
      PRINT(getPointer(deviceID));
      BARNEY_CUDA_CALL(Memcpy((void*)(getPointer(deviceID)+materialID),
                              &dd,sizeof(dd),cudaMemcpyDefault));
    }



    SamplerLibrary::SamplerLibrary(DevGroup::SP devGroup)
      : devGroup(devGroup)
    {
      PING;
      numReserved = 1;
      buffer = owlDeviceBufferCreate
        (devGroup->owl,OWL_USER_TYPE(Sampler::DD),1,nullptr);
    }

    SamplerLibrary::~SamplerLibrary()
    {
      PING;
      owlBufferRelease(buffer);
    }
    
    void SamplerLibrary::grow()
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

    int SamplerLibrary::allocate()
    {
      if (!reusableIDs.empty()) {
        int ID = reusableIDs.top();
        reusableIDs.pop();
        return ID;
      }
      if (nextFree == numReserved) grow();

      return nextFree++;
    }
   
    void SamplerLibrary::release(int nowReusableID)
    {
      reusableIDs.push(nowReusableID);
    }
  
    const Sampler::DD *SamplerLibrary::getPointer(int owlDeviceID) const
    {
      return (Sampler::DD *)owlBufferGetPointer(buffer,owlDeviceID);
    }    

    
  }
}
