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
#include "barney/DeviceGroup.h"

namespace BARNEY_NS {

  rtc::DataType toRTC(BNDataType type)
  {
    switch (type) {
    case BN_FLOAT32:      return rtc::FLOAT;
    case BN_FLOAT32_VEC2: return rtc::FLOAT2;
    case BN_FLOAT32_VEC3: return rtc::FLOAT3;
    case BN_FLOAT32_VEC4: return rtc::FLOAT4;
      
    case BN_FLOAT64:      
    case BN_FLOAT64_VEC2: 
    case BN_FLOAT64_VEC3: 
    case BN_FLOAT64_VEC4:
      throw std::runtime_error("rtc does not yet support doubles");
      
    case BN_INT32:
    case BN_UINT32:
      return rtc::INT;
      
    case BN_INT32_VEC2:
    case BN_UINT32_VEC2:
      return rtc::INT2;
      
    case BN_INT32_VEC3:
    case BN_UINT32_VEC3:
      return rtc::INT3;
      
    case BN_INT32_VEC4:
    case BN_UINT32_VEC4:
      return rtc::INT4;
      


    case BN_INT64:
      return rtc::LONG;
      
    case BN_INT64_VEC2:
      return rtc::LONG2;
      
    case BN_INT64_VEC3:
      return rtc::LONG3;
      
    case BN_INT64_VEC4:
      return rtc::LONG4;
      
    case BN_UFIXED8:
      return rtc::UCHAR;
      
    case BN_UFIXED16:
      return rtc::USHORT;
      
    case BN_UFIXED8_RGBA:
      return rtc::UCHAR4;
      
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

    case BN_INT8:      return "BN_INT8";
    case BN_INT8_VEC2: return "BN_INT8_VEC2";
    case BN_INT8_VEC3: return "BN_INT8_VEC3";
    case BN_INT8_VEC4: return "BN_INT8_VEC4";

    case BN_UINT8:      return "BN_UINT8";
    case BN_UINT8_VEC2: return "BN_UINT8_VEC2";
    case BN_UINT8_VEC3: return "BN_UINT8_VEC3";
    case BN_UINT8_VEC4: return "BN_UINT8_VEC4";

    case BN_INT32:      return "BN_INT32";
    case BN_INT32_VEC2: return "BN_INT32_VEC2";
    case BN_INT32_VEC3: return "BN_INT32_VEC3";
    case BN_INT32_VEC4: return "BN_INT32_VEC4";

    case BN_UINT32:      return "BN_UINT32";
    case BN_UINT32_VEC2: return "BN_UINT32_VEC2";
    case BN_UINT32_VEC3: return "BN_UINT32_VEC3";
    case BN_UINT32_VEC4: return "BN_UINT32_VEC4";

    case BN_INT64:      return "BN_INT64";
    case BN_INT64_VEC2: return "BN_INT64_VEC2";
    case BN_INT64_VEC3: return "BN_INT64_VEC3";
    case BN_INT64_VEC4: return "BN_INT64_VEC4";

    case BN_UINT64:      return "BN_UINT64";
    case BN_UINT64_VEC2: return "BN_UINT64_VEC2";
    case BN_UINT64_VEC3: return "BN_UINT64_VEC3";
    case BN_UINT64_VEC4: return "BN_UINT64_VEC4";

    case BN_FLOAT32:      return "BN_FLOAT32";
    case BN_FLOAT32_VEC2: return "BN_FLOAT32_VEC2";
    case BN_FLOAT32_VEC3: return "BN_FLOAT32_VEC3";
    case BN_FLOAT32_VEC4: return "BN_FLOAT32_VEC4";

    case BN_FLOAT64:      return "BN_FLOAT64";
    case BN_FLOAT64_VEC2: return "BN_FLOAT64_VEC2";
    case BN_FLOAT64_VEC3: return "BN_FLOAT64_VEC3";
    case BN_FLOAT64_VEC4: return "BN_FLOAT64_VEC4";

    default:
      throw std::runtime_error
        ("#bn internal error: to_string not implemented for "
         "numerical BNDataType #"+std::to_string(int(type)));
    };
  }

  std::string to_string(BNFrameBufferChannel channel)
  {
    switch(channel) {
    case BN_FB_COLOR:  return "BN_FB_COLOR";
    case BN_FB_DEPTH:  return "BN_FB_DEPTH";
    case BN_FB_PRIMID: return "BN_FB_PRIMID";
    case BN_FB_INSTID: return "BN_FB_INSTID";
    case BN_FB_OBJID:  return "BN_FB_OBJID";
    default:
      throw std::runtime_error
        ("#bn internal error: to_string not implemented for "
         "numerical BNFrameBufferChannel #"+std::to_string(int(channel)));
    };
  }
  
  
  size_t owlSizeOf(BNDataType type)
  {
    switch (type) {
    case BN_FLOAT32:      return sizeof(float);
    case BN_FLOAT32_VEC2: return sizeof(vec2f);
    case BN_FLOAT32_VEC3: return sizeof(vec3f);
    case BN_FLOAT32_VEC4: return sizeof(vec4f);

    case BN_INT8:      return sizeof(int8_t);
    case BN_INT8_VEC2: return sizeof(vec2c);
    case BN_INT8_VEC3: return sizeof(vec3c);
    case BN_INT8_VEC4: return sizeof(vec4c);

    case BN_UINT8:      return sizeof(uint8_t);
    case BN_UINT8_VEC2: return sizeof(vec2uc);
    case BN_UINT8_VEC3: return sizeof(vec3uc);
    case BN_UINT8_VEC4: return sizeof(vec4uc);

    case BN_INT32:      return sizeof(int32_t);
    case BN_INT32_VEC2: return sizeof(vec2i);
    case BN_INT32_VEC3: return sizeof(vec3i);
    case BN_INT32_VEC4: return sizeof(vec4i);

    case BN_UINT32:      return sizeof(uint32_t);
    case BN_UINT32_VEC2: return sizeof(vec2ui);
    case BN_UINT32_VEC3: return sizeof(vec3ui);
    case BN_UINT32_VEC4: return sizeof(vec4ui);

    case BN_INT64:
      return sizeof(int64_t);
    case BN_INT64_VEC2:
      return sizeof(vec2l);
    case BN_INT64_VEC3:
      return sizeof(vec3l);
    case BN_INT64_VEC4:
      return sizeof(vec4l);

    case BN_UINT64:
      return sizeof(uint64_t);
    case BN_UINT64_VEC2:
      return sizeof(vec2ul);
    case BN_UINT64_VEC3:
      return sizeof(vec3ul);
    case BN_UINT64_VEC4:
      return sizeof(vec4ul);
      
    default:
      throw std::runtime_error
        ("#bn internal error: owlSizeOf() not implemented for "
         "numerical BNDataType #"+std::to_string(int(type)));
    };
  }

  BaseData::BaseData(Context *context,
             const DevGroup::SP &devices,
             BNDataType type)
    : barney_api::Data(context),
      type(type),
      count(0),
      devices(devices)
  {}


  void PODData::set(const void *_items, int count)
  {
    this->count = count;
    for (auto device : *devices) {
      getPLD(device)->rtcBuffer->resize(count*owlSizeOf(type));
      getPLD(device)->rtcBuffer->upload(_items,count*owlSizeOf(type));
    }
  }

  PODData::PODData(Context *context,
                   const DevGroup::SP &devices,
                   BNDataType type)
    : BaseData(context,devices,type)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      getPLD(device)->rtcBuffer 
        = device->rtc->createBuffer(1);//*owlSizeOf(type),_items);
      assert(getPLD(device)->rtcBuffer);
    }
    for (auto device : *devices)
      device->sync();
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
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }
  
  BaseData::SP BaseData::create(Context *context,
                                const DevGroup::SP &devices,
                                BNDataType type)
  {
    switch(type) {
    case BN_INT8:
    case BN_INT8_VEC2:
    case BN_INT8_VEC3:
    case BN_INT8_VEC4:
    case BN_UINT8:
    case BN_UINT8_VEC2:
    case BN_UINT8_VEC3:
    case BN_UINT8_VEC4:
    case BN_INT32:
    case BN_INT32_VEC2:
    case BN_INT32_VEC3:
    case BN_INT32_VEC4:
    case BN_UINT32:
    case BN_UINT32_VEC2:
    case BN_UINT32_VEC3:
    case BN_UINT32_VEC4:
    case BN_INT64:
    case BN_INT64_VEC2:
    case BN_INT64_VEC3:
    case BN_INT64_VEC4:
    case BN_UINT64:
    case BN_UINT64_VEC2:
    case BN_UINT64_VEC3:
    case BN_UINT64_VEC4:
    case BN_FLOAT32:
    case BN_FLOAT32_VEC2:
    case BN_FLOAT32_VEC3:
    case BN_FLOAT32_VEC4:
    case BN_FLOAT64:
    case BN_FLOAT64_VEC2:
    case BN_FLOAT64_VEC3:
    case BN_FLOAT64_VEC4:
      return std::make_shared<PODData>
        (context,devices,type);
    case BN_OBJECT:
      return std::make_shared<ObjectRefsData>
        (context,devices,type);
    default:
      throw std::runtime_error("un-implemented data type '"
                               +to_string(type)
                               +" in Data::create()");
    }
  }
  
  ObjectRefsData::ObjectRefsData(Context *context,
                                 const DevGroup::SP &devices,
                                 BNDataType type)
    : BaseData(context,devices,type)
  {
    // items.resize(numItems);
    // for (int i=0;i<numItems;i++)
    //   items[i] = (((Object **)_items)[i])->shared_from_this();
  }

  void ObjectRefsData::set(const void *_items, int count)
  {
    this->count = count;
    items.resize(count);
    for (int i=0;i<count;i++)
      items[i] = (((Object **)_items)[i])->shared_from_this();
  }


}
