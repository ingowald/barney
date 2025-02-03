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

#include "barney/common/Data.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

namespace barney {
  
  rtc::DataType toRTC(BNDataType type)
  {
    switch (type) {
    case BN_FLOAT:
      return rtc::FLOAT;
      
    case BN_FLOAT2:
      return rtc::FLOAT2;
      
    case BN_FLOAT3:
      return rtc::FLOAT3;
      
    case BN_FLOAT4:
      return rtc::FLOAT4;
      
    case BN_INT:
      return rtc::INT;
      
    case BN_INT2:
      return rtc::INT2;
      
    case BN_INT3:
      return rtc::INT3;
      
    case BN_INT4:
      return rtc::INT4;
      
    case BN_UFIXED8:
      return rtc::UCHAR;
      
    case BN_UFIXED16:
      return rtc::USHORT;
      
    case BN_UFIXED8_RGBA:
      return rtc::UCHAR4;
      
    case BN_FLOAT4_RGBA:
      return rtc::FLOAT4;
      
    default: throw std::runtime_error
        ("un-recognized barney data type #"
         +std::to_string((int)type));
    };
  }
  
  std::string to_string(BNDataType type)
  {
    switch (type) {
    case BN_DATA_UNDEFINED:
      return "BN_DATA_UNDEFINED";
    case BN_DATA:
      return "BN_DATA";
    case BN_OBJECT:
      return "BN_OBJECT";
    case BN_TEXTURE:
      return "BN_TEXTURE";
    case BN_INT:
      return "BN_INT";
    case BN_INT2:
      return "BN_INT2";
    case BN_INT3:
      return "BN_INT3";
    case BN_INT4:
      return "BN_INT4";
    case BN_FLOAT:
      return "BN_FLOAT";
    case BN_FLOAT2:
      return "BN_FLOAT2";
    case BN_FLOAT3:
      return "BN_FLOAT3";
    case BN_FLOAT4:
      return "BN_FLOAT4";
    default:
      throw std::runtime_error
        ("#bn internal error: to_string not implemented for "
         "numerical BNDataType #"+std::to_string(int(type)));
    };
  };
  
  size_t owlSizeOf(BNDataType type)
  {
    switch (type) {
    case BN_FLOAT:
      return sizeof(float);
    case BN_FLOAT2:
      return sizeof(vec2f);
    case BN_FLOAT3:
      return sizeof(vec3f);
    case BN_FLOAT4:
      return sizeof(vec4f);
    case BN_INT:
      return sizeof(int);
    case BN_INT2:
      return sizeof(vec2i);
    case BN_INT3:
      return sizeof(vec3i);
    case BN_INT4:
      return sizeof(vec4i);
    default:
      throw std::runtime_error
        ("#bn internal error: owlSizeOf() not implemented for "
         "numerical BNDataType #"+std::to_string(int(type)));
    };
  };
  
  Data::Data(Context *context,
             const DevGroup::SP &devices,
             BNDataType type,
             size_t numItems)
    : SlottedObject(context,devices),
      type(type),
      count(numItems)
  {}


  PODData::PODData(Context *context,
                   const DevGroup::SP &devices,
                   BNDataType type,
                   size_t numItems,
                   const void *_items)
    : Data(context,devices,type,numItems)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      getPLD(device)->rtcBuffer 
        = device->rtc->createBuffer(numItems*owlSizeOf(type),_items);
      assert(getPLD(device)->rtcBuffer);
    }
  }
  
  PODData::~PODData()
  {
    for (auto device : *devices)
      device->rtc->freeBuffer(getPLD(device)->rtcBuffer);
  }

  const void *PODData::getDD(Device *device) 
  {
    assert(device);
    PLD *pld = getPLD(device);
    assert(pld);
    assert(pld->rtcBuffer);
    return pld->rtcBuffer->getDD();
  }
  
  PODData::PLD *PODData::getPLD(Device *device) 
  {
    assert(device);
    assert(device->contextRank >= 0);
    assert(device->contextRank < perLogical.size());
    return &perLogical[device->contextRank];
  }
  
  Data::SP Data::create(Context *context,
                        const DevGroup::SP &devices,
                        BNDataType type,
                        size_t numItems,
                        const void *items)
  {
    switch(type) {
    case BN_INT:
    case BN_INT2:
    case BN_INT3:
    case BN_INT4:
    case BN_FLOAT:
    case BN_FLOAT2:
    case BN_FLOAT3:
    case BN_FLOAT4:
      return std::make_shared<PODData>
        (context,devices,type,numItems,items);
    case BN_OBJECT:
      return std::make_shared<ObjectRefsData>
        (context,devices,type,numItems,items);
    default:
      throw std::runtime_error("un-implemented data type '"
                               +to_string(type)
                               +" in Data::create()");
    }
  }
  
  ObjectRefsData::ObjectRefsData(Context *context,
                                 const DevGroup::SP &devices,
                                 BNDataType type,
                                 size_t numItems,
                                 const void *_items)
    : Data(context,devices,type,numItems)
  {
    items.resize(numItems);
    for (int i=0;i<numItems;i++)
      items[i] = (((Object **)_items)[i])->shared_from_this();
  }
};
