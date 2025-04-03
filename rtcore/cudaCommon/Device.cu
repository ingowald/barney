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
    
  }
}

