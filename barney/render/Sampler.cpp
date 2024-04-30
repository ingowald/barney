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

#include "barney/render/Sampler.h"
#include "barney/ModelSlot.h"

namespace barney {
  namespace render {

    Sampler::Sampler(ModelSlot *owner)
      : SlottedObject(owner),
        samplerID(owner->world.samplerLibrary.allocate())
    {}
    
    Sampler::~Sampler()
    {
      owner->world.samplerLibrary.release(samplerID);
    }
    
    Sampler::SP Sampler::create(ModelSlot *owner, const std::string &type)
    {
      PING; PRINT(type);
      if (type == "image1D")
        return std::make_shared<ImageSampler>(owner,1);
      if (type == "image2D")
        return std::make_shared<ImageSampler>(owner,2);
      if (type == "image3D")
        return std::make_shared<ImageSampler>(owner,3);
      if (type == "transform")
        return std::make_shared<TransformSampler>(owner);
      throw std::runtime_error("not implemented");
    }

    void ImageSampler::createDD(DD &dd, int devID)
    {
      PING; throw std::runtime_error("not implemneted");
    }
    
    void TransformSampler::createDD(DD &dd, int devID)
    {
      PING; throw std::runtime_error("not implemneted");
    }

  }
}
