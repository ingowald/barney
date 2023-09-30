// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney.h"
#include "mori/DeviceGroup.h"
#include <string.h>
#include <cuda_runtime.h>
#include <mutex>
#include <map>

namespace barney {
  using namespace owl::common;
  
  struct Object {
    typedef std::shared_ptr<Object> SP;
    
    virtual std::string toString() const { return "<Object>"; }
  };
  
  struct FrameBuffer : public Object {
    virtual void resize(vec2i size) = 0;
  };

  struct Context {

    virtual FrameBuffer *createFB() = 0;
    
    template<typename T>
    T *initReference(std::shared_ptr<T> sp)
    {
      std::lock_guard<std::mutex> lock(mutex);
      hostOwnedHandles[sp]++;
      return sp.get();
    }

    Context(const std::vector<int> &dataGroupIDs,
            const std::vector<int> &gpuIDs);
            
    const std::vector<int> dataGroupIDs;
    const std::vector<int> gpuIDs;
            
    std::mutex mutex;
    std::map<Object::SP,int> hostOwnedHandles;
    std::vector<mori::DeviceGroup::SP> moris;
  };


  
  struct LocalFB : public FrameBuffer {
    typedef std::shared_ptr<LocalFB> SP;

    static SP create()
    { return std::make_shared<LocalFB>(); }
    
    void resize(vec2i size) override { PING; }
  };
  
  struct LocalContext : public Context {
    LocalContext(const std::vector<int> &dataGroupIDs,
                 const std::vector<int> &gpuIDs)
      : Context(dataGroupIDs,gpuIDs)
    {}

    FrameBuffer *createFB() override
    { return initReference(LocalFB::create()); }
  };
  
}

