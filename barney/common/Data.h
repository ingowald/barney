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

#include "barney/Object.h"

namespace barney {

  rtc::DataType toRTC(BNDataType format);
  
  std::string to_string(BNDataType type);
  size_t owlSizeOf(BNDataType type);
  
  struct Data : public SlottedObject {
    typedef std::shared_ptr<Data> SP;

    Data(Context *context,
         const DevGroup::SP &devices,
         BNDataType type,
         size_t numItems);
    virtual ~Data() = default;
    
    static Data::SP create(Context *context,
                           const DevGroup::SP &devices,
                           BNDataType type,
                           size_t numItems,
                           const void *items);
    
    BNDataType type  = BN_DATA_UNDEFINED;
    size_t     count = 0;
  };

  /*! a data array for 'plain-old-data' type data (such as int, float,
      vec3f, etc) that does not need reference-counting for object
      lifetime handling; class-type data (any BNWhatEver) needs to be
      in ObecjtsRefData which does the refcounting */
  struct PODData : public Data {
    typedef std::shared_ptr<PODData> SP;
    
    /*! constructor for a 'global' data array that lives on the
        context itself, and spans all model slots */
    PODData(Context *context,
            const DevGroup::SP &devices,
            BNDataType type,
            size_t numItems,
            const void *items);
    
    virtual ~PODData();

    const void *getDD(Device *device) 
    { return getPLD(device)->rtcBuffer->getDD(); }

    struct PLD {
      rtc::Buffer *rtcBuffer   = 0;
    };
    PLD *getPLD(Device *device) 
    { return &perLogical[device->contextRank]; }
    std::vector<PLD> perLogical;
  };

  /*! data array over reference-counted barney object handles (e.g.,
      BNTexture's, BNlight's, etc. this has to make sure that objects
      put into this data array will remain properly refcoutned. */
  struct ObjectRefsData : public Data {
    typedef std::shared_ptr<ObjectRefsData> SP;
    ObjectRefsData(Context *context,
                   const DevGroup::SP &devices,
                   BNDataType type,
                   size_t numItems,
                   const void *items);
    std::vector<Object::SP> items;
  };

}
