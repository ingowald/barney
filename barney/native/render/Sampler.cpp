// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/Sampler.h"
#include "native/common/Texture.h"
#include "native/ModelSlot.h"
#include "native/render/SamplerRegistry.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    AttributeKind parseAttribute(const std::string &attributeName)
    {
      if (attributeName == "attribute0")
        return ATTRIBUTE_0; 
      if (attributeName == "attribute1")
        return ATTRIBUTE_1; 
      if (attributeName == "attribute2")
        return ATTRIBUTE_2; 
      if (attributeName == "attribute3")
        return ATTRIBUTE_3; 
      if (attributeName == "worldPosition")
        return WORLD_POSITION; 
      if (attributeName == "objectPosition")
        return OBJECT_POSITION; 
      if (attributeName == "color")
        return COLOR; 
      
      throw std::runtime_error("@barney: invalid or unsupported attribute name '"
                               +attributeName+"'");
    }

    Sampler::Sampler(Device *device)
      : Object(device),
        samplerRegistry(device->samplerRegistry),
        samplerID(device->samplerRegistry->allocate())
    {}
    
    Sampler::~Sampler()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~Sampler deconstructing"
                     << std::endl);
      samplerRegistry->release(samplerID);
    }

    Sampler::SP Sampler::create(Device *device,
                                const std::string &type)
    {
      if (type == "texture1D" || type == "image1D")
        return std::make_shared<TextureSampler>(device,1);
      if (type == "texture2D" || type == "image2D")
        return std::make_shared<TextureSampler>(device,2);
      if (type == "texture3D" || type == "image3D")
        return std::make_shared<TextureSampler>(device,3);
      if (type == "transform")
        return std::make_shared<TransformSampler>(device);
      if (type == "primitive")
        return std::make_shared<PrimitiveSampler>(device);
      throw std::runtime_error("do not know what a '"+type+" sampler is !?");
    }


    bool Sampler::setObject(const std::string &member,
                            const std::shared_ptr<Object> &value)
    {
      return false;
    }
        
    bool Sampler::setString(const std::string &member,
                            const std::string &value)
    {
      if (member == "inAttribute")
        { inAttribute = parseAttribute(value); return true; }
      
      return false;
    }

    bool Sampler::set4x4f(const std::string &member, const vec4f *value)
    {
      if (member == "outTransform")
        { outTransform = *(mat4f*)value; return true; }
      
      return false;
    }
    
    bool Sampler::set4f(const std::string &member, const vec4f &value)
    {
      if (member == "outOffset")
        { outOffset = value; return true; }
      if (member == "borderColor")
        { borderColor = value; return true; }

      return false;
    }

    void Sampler::commit() 
    {
      // for (auto device : *devices) {
      DD dd = getDD();
      samplerRegistry->setDD(samplerID,dd);
      // }
    }

    TextureSampler::TextureSampler(Device *device,
                                   int numDims)
      : Sampler(device),
        numDims(numDims)
    {
      // perLogical.resize(devices->numLogical);
    }
    
    TextureSampler::~TextureSampler()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~TextureSampler deconstructing"
                     << std::endl);
      // for (auto device : *devices) {
      //   PLD *pld = getPLD(device);
      if (rtcTexture)
        device->rtc->freeTexture(rtcTexture);
      rtcTexture = 0;
    }
    
    bool TextureSampler::setObject(const std::string &member,
                                   const std::shared_ptr<Object> &value)
    {
      if (Sampler::setObject(member,value)) return true;

      if (member == "textureData") {
        textureData = value ? value->as<TextureData>() : TextureData::SP();
        return true;
      }
      
      return false;
    }

    bool TextureSampler::set1i(const std::string &member, const int   &value) 
    {
      if (Sampler::set1i(member,value)) return true;

      if (member == "wrapMode0")
        { wrapModes[0] = (BNTextureAddressMode)value; return true; }      
      if (member == "wrapMode1")
        { wrapModes[1] = (BNTextureAddressMode)value; return true; }      
      if (member == "wrapMode2")
        { wrapModes[2] = (BNTextureAddressMode)value; return true; }      
      if (member == "filterMode")
        { filterMode = (BNTextureFilterMode)value; return true; }      
      if (member == "colorSpace")
        { colorSpace = (BNTextureColorSpace)value; return true; }      

      return false;
    }
    
    bool TextureSampler::set4x4f(const std::string &member, const vec4f *value)
    {
      if (Sampler::set4x4f(member,value)) return true;

      if (member == "inTransform")
        { inTransform = *(mat4f*)value; return true; }      

      return false;
    }
    
    bool TextureSampler::set4f(const std::string &member, const vec4f &value)
    {
      if (Sampler::set4f(member,value)) return true;

      if (member == "inOffset")
        { inOffset = value; return true; }
      
      return false;
    }

    // TextureSampler::PLD *TextureSampler::getPLD(Device *device) 
    // {
    //   assert(device);
    //   assert(device->contextRank() >= 0);
    //   assert(device->contextRank() < perLogical.size());
    //   return &perLogical[device->contextRank()];
    // }
    
    void TextureSampler::commit() 
    {
      // for (auto device : *devices) {
      //   PLD *pld = getPLD(device);
      if (rtcTexture) {
        device->rtc->freeTexture(rtcTexture);
        rtcTexture = 0;
      }

      if (!textureData) {
        std::cerr << "WARNING: Image Sampler without any texture data?"
                  << std::endl;
      } else {
        rtc::TextureDesc desc;
        desc.filterMode = toRTC(filterMode);
        desc.addressMode[0] = toRTC(wrapModes[0]);
        desc.addressMode[1] = toRTC(wrapModes[1]);
        desc.addressMode[2] = toRTC(wrapModes[2]);
        desc.colorSpace     = toRTC(colorSpace);
        desc.borderColor    = borderColor;
        // for (auto device : *devices) {
        //   PLD *pld = getPLD(device);
        //   if (pld->rtcTexture)
        //     device->rtc->freeTexture(pld->rtcTexture);
        rtcTexture
          = textureData->rtc->createTexture(desc);//getPLD(device)->rtc->createTexture(desc);
        // };
      }

      Sampler::commit();
    }

    Sampler::DD TextureSampler::getDD() 
    {
      Sampler::DD dd;
      switch(numDims) {
      case 1:
        dd.type = Sampler::IMAGE1D; break;
      case 2:
        dd.type = Sampler::IMAGE2D; break;
      case 3:
        dd.type = Sampler::IMAGE3D; break;
      };
      dd.inAttribute = (AttributeKind)inAttribute;

      (vec4f&)dd.outTransform.offset = outOffset;
      memcpy(&dd.outTransform.mat_x,&outTransform,sizeof(outTransform));
      
      (vec4f&)dd.inTransform.offset = inOffset;
      memcpy(&dd.inTransform.mat_x,&inTransform,sizeof(inTransform));

      // PLD *pld = getPLD(device);
      if (!rtcTexture) {
        std::cout << "WARN: NO TEXTURE DATA ON IMAGE SAMPLER!" << std::endl;
        dd.texture = 0;
      } else {
        dd.texture = rtcTexture->getDD();
        dd.numChannels = textureData->numChannels;
      }
      return dd;
    }
    
    Sampler::DD TransformSampler::getDD() 
    {
      Sampler::DD dd;
      dd.type = Sampler::TRANSFORM;
      dd.inAttribute = (AttributeKind)inAttribute;
      (vec4f&)dd.outTransform.offset = outOffset;
      memcpy(&dd.outTransform.mat_x,&outTransform,sizeof(outTransform));
      return dd;
    }




    PrimitiveSampler::PrimitiveSampler(Device *device)
      : Sampler(device)
    {
    }
    
    PrimitiveSampler::~PrimitiveSampler()
    {}
    
    bool PrimitiveSampler::setData(const std::string &member,
                                   const std::shared_ptr<Data> &value)
    {
      if (Sampler::setObject(member,value)) return true;

      if (member == "arrayData") {
        arrayData = value ? value->as<PODData>() : PODData::SP();
        return true;
      }
      
      return false;
    }

    bool PrimitiveSampler::set1i(const std::string &member, const int   &value) 
    {
      if (Sampler::set1i(member,value)) return true;

      if (member == "arrayOffset")
        { arrayOffset = value; return true; }      
      if (member == "arrayType")
        { arrayType = (BNDataType)value; return true; }      

      return false;
    }
    
    Sampler::DD PrimitiveSampler::getDD() 
    {
      Sampler::DD dd;
      dd.type = Sampler::PRIMITIVE;
      dd.inAttribute = (AttributeKind)inAttribute;

      (vec4f&)dd.outTransform.offset = outOffset;
      memcpy(&dd.outTransform.mat_x,&outTransform,sizeof(outTransform));

      dd.arrayData
        = arrayData
        ? arrayData->rtcBuffer->getDD()
        : 0;
      dd.arrayOffset = arrayOffset;
      dd.arraySize = arrayData ? (int)arrayData->count : 0;
      dd.arrayType = arrayType;
      
      return dd;
    }


    TransformSampler::TransformSampler(Device *device)
      : Sampler(device)
    {}
    
    std::string TransformSampler::toString() const 
    { return "TransformSampler"; }
    
  }
}
