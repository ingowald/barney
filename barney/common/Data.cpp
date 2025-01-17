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

  const std::string to_string(BNDataType type)
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
      throw std::runtime_error("#bn internal error: to_string not implemented for "
                               "numerical BNDataType #"+std::to_string(int(type)));
    };
  };
  
  OWLDataType owlTypeFor(BNDataType type)
  {
    switch (type) {
    case BN_FLOAT:
      return OWL_FLOAT;
    case BN_FLOAT2:
      return OWL_FLOAT2;
    case BN_FLOAT3:
      return OWL_FLOAT3;
    case BN_FLOAT4:
      return OWL_FLOAT4;
    case BN_INT:
      return OWL_INT;
    case BN_INT2:
      return OWL_INT2;
    case BN_INT3:
      return OWL_INT3;
    case BN_INT4:
      return OWL_INT4;
    default:
      throw std::runtime_error("#bn internal error: owlTypeFor() not implemented for "
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
      throw std::runtime_error("#bn internal error: owlSizeOf() not implemented for "
                               "numerical BNDataType #"+std::to_string(int(type)));
    };
  };
  
  PODData::PODData(Context *context,
                   int slot,
                   BNDataType type,
                   size_t numItems,
                   const void *_items)
    : Data(context,slot,type,numItems)
  {
    // owl = owlDeviceBufferCreate(getOWL(),owlTypeFor(type),
    //                             numItems,_items);
    rtcBuffer
      = getRTC()->createBuffer(numItems*owlSizeOf(type),_items);
  }

  PODData::~PODData()
  {
    if (rtcBuffer) getRTC()->free(rtcBuffer);
    rtcBuffer = 0;
    // if (owl) owlBufferRelease(owl);
  }

  Data::SP Data::create(Context *context,
                        int slot,
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
      return std::make_shared<PODData>(context,slot,type,numItems,items);
    case BN_OBJECT:
      return std::make_shared<ObjectRefsData>(context,slot,type,numItems,items);
    default:
      throw std::runtime_error("un-implemented data type '"+to_string(type)
                               +" in Data::create()");
    }
  }
  
  Data::Data(Context *context,
             int slot,
             BNDataType type,
             size_t numItems)
    : SlottedObject(context,slot),
      type(type),
      count(numItems)
  {}

  ObjectRefsData::ObjectRefsData(Context *context,
                                 int slot,
                                 BNDataType type,
                                 size_t numItems,
                                 const void *_items)
    : Data(context,slot,type,numItems)
  {
    items.resize(numItems);
    for (int i=0;i<numItems;i++)
      items[i] = (((Object **)_items)[i])->shared_from_this();
  }
};
