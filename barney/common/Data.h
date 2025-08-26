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

namespace BARNEY_NS {

  rtc::DataType toRTC(BNDataType format);
  
  std::string to_string(BNDataType type);
  std::string to_string(BNFrameBufferChannel type);
  size_t owlSizeOf(BNDataType type);
  
  struct BaseData : public barney_api::Data {//SlottedObject {
    typedef std::shared_ptr<Data> SP;

    BaseData(Context *context,
             const DevGroup::SP &devices,
             BNDataType type);
    virtual ~BaseData() = default;
    
    static BaseData::SP create(Context *context,
                               const DevGroup::SP &devices,
                               BNDataType type);

    BNDataType type  = BN_DATA_UNDEFINED;
    size_t     count = 0;
    DevGroup::SP const devices;
  };

  /*! a data array for 'plain-old-data' type data (such as int, float,
      vec3f, etc) that does not need reference-counting for object
      lifetime handling; class-type data (any BNWhatEver) needs to be
      in ObecjtsRefData which does the refcounting */
  struct PODData : public BaseData {
    typedef std::shared_ptr<PODData> SP;
    
    /*! constructor for a 'global' data array that lives on the
        context itself, and spans all model slots */
    PODData(Context *context,
            const DevGroup::SP &devices,
            BNDataType type);
    
    virtual ~PODData();

    const void *getDD(Device *device);
    void set(const void *data, int count) override;

    struct PLD {
      rtc::Buffer *rtcBuffer   = 0;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
  };

  /*! data array over reference-counted barney object handles (e.g.,
      BNTexture's, BNlight's, etc. this has to make sure that objects
      put into this data array will remain properly refcoutned. */
  struct ObjectRefsData : public BaseData {
    typedef std::shared_ptr<ObjectRefsData> SP;
    ObjectRefsData(Context *context,
                   const DevGroup::SP &devices,
                   BNDataType type);
    void set(const void *data, int count) override;
    std::vector<Object::SP> items;
  };

}
