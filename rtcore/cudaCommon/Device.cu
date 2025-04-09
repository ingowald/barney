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

#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cudaCommon/Texture.h"
#include "rtcore/cudaCommon/TextureData.h"

namespace rtc {
  namespace cuda_common {

    int Device::setActive() const
    {
      int oldActive = 0;
      BARNEY_CUDA_CHECK(cudaGetDevice(&oldActive));
      BARNEY_CUDA_CHECK(cudaSetDevice(physicalID));
      return oldActive;
    }
    
    void Device::restoreActive(int oldActive) const
    {
      BARNEY_CUDA_CHECK(cudaSetDevice(oldActive));
    }
    
    void *Device::allocMem(size_t numBytes)
    {
      if (!numBytes) return nullptr;
      SetActiveGPU forDuration(this);
      void *ptr = 0;
        BARNEY_CUDA_CALL(Malloc((void **)&ptr,numBytes));
        assert(ptr);
      return ptr;
    }
    
    void *Device::allocHost(size_t numBytes) 
    {
      if (!numBytes) return nullptr;
      SetActiveGPU forDuration(this);
      void *ptr = 0;
      BARNEY_CUDA_CALL(MallocHost(&ptr,numBytes));
      return ptr;
    }
      
    void Device::freeHost(void *mem) 
    {
      if (!mem) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(FreeHost(mem));
    }
      
    void Device::freeMem(void *mem) 
    {
      if (!mem) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(Free(mem));
    }
      
    void Device::memsetAsync(void *mem,int value, size_t numBytes) 
    {
      if (numBytes == 0) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(MemsetAsync(mem,value,numBytes,stream));
    }
      

    void Device::copyAsync(void *dst, const void *src, size_t numBytes) 
    {
      if (numBytes == 0) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(MemcpyAsync(dst,src,numBytes,cudaMemcpyDefault,stream));
    }
      
    void Device::sync() 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(StreamSynchronize(stream));
    }

    void Device::freeTextureData(TextureData *td)
    {
      if (td) delete td;
    }
    
    void Device::freeTexture(Texture *tex)
    {
      if (tex) delete tex;
    }
    
    TextureData *
    Device::createTextureData(vec3i dims,
                              rtc::DataType format,
                              const void *texels) 
    {
      SetActiveGPU forDuration(this);
      return new TextureData(this,dims,format,texels);
    }

    Texture *TextureData::createTexture(const TextureDesc &desc) 
    {
      SetActiveGPU forDuration(device);
      return new Texture(this,desc);
    }    

    /*! enable peer access between these gpus, and return truea if
        successful, else if at least one pair does not work */
    bool enablePeerAccess(const std::vector<int> &gpuIDs)
    {
#define LOG(a) ss << "#bn." << a << std::endl;

      std::stringstream ss;
      ss << "enabling peer access ('.'=self, '+'=can access other device)" << std::endl;
 
     
      int deviceCount = (int)gpuIDs.size();
      LOG("found " << deviceCount << " CUDA capable devices");
      for (auto gpuID : gpuIDs) {
        cudaDeviceProp prop;
        BARNEY_CUDA_CALL(GetDeviceProperties(&prop, gpuID));
        LOG(" - device #" << gpuID << " : " << prop.name);
      }
      LOG("enabling peer access:");
      
      bool successful = true;
      for (auto gpuID : gpuIDs) {
        std::stringstream ss;
        SetActiveGPU forLifeTime(gpuID);
        ss << " - device #" << gpuID << " : ";
        int cuda_i = gpuID;
        int i = gpuID;
        for (int j=0;j<deviceCount;j++) {
          if (j == i) {
            ss << " ."; 
          } else {
            int cuda_j = gpuIDs[j];
            int canAccessPeer = 0;
            cudaError_t rc = cudaDeviceCanAccessPeer(&canAccessPeer, cuda_i,cuda_j);
            if (rc != cudaSuccess)
              throw std::runtime_error("cuda error in cudaDeviceCanAccessPeer: "
                                       +std::to_string(rc));
            if (!canAccessPeer) {
              // huh. this can happen if you have differnt device
              // types (in my case, a 2070 and a rtx 8000).
              // nvm - yup, this isn't an error. Expect certain configs to not allow peer access.
              // disabling this, as it's concerning end users.
              // std::cerr << "cannot not enable peer access!? ... skipping..." << std::endl;
              successful = false;
              continue;
            }
            
            rc = cudaDeviceEnablePeerAccess(cuda_j,/* flags - must be 0 */0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) {
              cudaGetLastError();
            } else if (rc != cudaSuccess)
              throw std::runtime_error("cuda error in cudaDeviceEnablePeerAccess: "
                                       +std::to_string(rc));
            ss << " +";
          }
        }
        LOG(ss.str());
      }
      return successful;
    }
  }
}

