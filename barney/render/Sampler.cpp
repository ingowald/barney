// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/render/Sampler.h"
#include "barney/common/Texture.h"
#include "barney/ModelSlot.h"
#include "barney/render/SamplerRegistry.h"
#include "barney/Context.h"

namespace BARNEY_NS {
  namespace render {
    
    AttributeKind parseAttribute(const std::string &attributeName)
    {
      if (attributeName == "attribute0")
        return render::ATTRIBUTE_0; 
      if (attributeName == "attribute1")
        return render::ATTRIBUTE_1; 
      if (attributeName == "attribute2")
        return render::ATTRIBUTE_2; 
      if (attributeName == "attribute3")
        return render::ATTRIBUTE_3; 
      if (attributeName == "worldPosition")
        return render::WORLD_POSITION; 
      if (attributeName == "objectPosition")
        return render::OBJECT_POSITION; 
      if (attributeName == "color")
        return render::COLOR; 
      
      throw std::runtime_error("@barney: invalid or unsupported attribute name '"
                               +attributeName+"'");
    }

    Sampler::Sampler(SlotContext *slotContext)
      : barney_api::Sampler(slotContext->context),
        devices(slotContext->devices),
        samplerRegistry(slotContext->samplerRegistry),
        samplerID(slotContext->samplerRegistry->allocate())
    {}
    
    Sampler::~Sampler()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~Sampler deconstructing"
                     << std::endl);
      samplerRegistry->release(samplerID);
    }

    Sampler::SP Sampler::create(SlotContext *context,
                                const std::string &type)
    {
      if (type == "texture1D" || type == "image1D")
        return std::make_shared<TextureSampler>(context,1);
      if (type == "texture2D" || type == "image2D")
        return std::make_shared<TextureSampler>(context,2);
      if (type == "texture3D" || type == "image3D")
        return std::make_shared<TextureSampler>(context,3);
      if (type == "transform")
        return std::make_shared<TransformSampler>(context);
      throw std::runtime_error("do not know what a '"+type+" sampler is !?");
    }


    bool Sampler::setObject(const std::string &member,
                                 const std::shared_ptr<Object> &value)
    {
      // if (SlottedObject::setObject(member,value)) return true;

      return false;
    }
        
    bool Sampler::setString(const std::string &member,
                            const std::string &value)
    {
      // if (SlottedObject::setString(member,value)) return true;

      if (member == "inAttribute")
        { inAttribute = parseAttribute(value); return true; }
      
      return false;
    }

    bool Sampler::set4x4f(const std::string &member, const vec4f *value)
    {
      // if (SlottedObject::set4x4f(member,value)) return true;

      if (member == "outTransform")
        { outTransform = *(mat4f*)value; return true; }
      
      return false;
    }
    
    bool Sampler::set4f(const std::string &member, const vec4f &value)
    {
      // if (SlottedObject::set4f(member,value)) return true;

      if (member == "outOffset")
        { outOffset = value; return true; }
      if (member == "borderColor")
        { borderColor = value; return true; }

      return false;
    }

    void Sampler::commit() 
    {
      // SlottedObject::commit();
      for (auto device : *devices) {
        DD dd = getDD(device);
        samplerRegistry->setDD(samplerID,dd,device);
      }
    }

    TextureSampler::TextureSampler(SlotContext *slotContext,
                                   int numDims)
      : Sampler(slotContext),
        numDims(numDims)
    {
      perLogical.resize(devices->numLogical);
    }
    
    TextureSampler::~TextureSampler()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~TextureSampler deconstructing"
                     << std::endl);
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        if (pld->rtcTexture)
          device->rtc->freeTexture(pld->rtcTexture);
        pld->rtcTexture = 0;
      }
    }
    
    bool TextureSampler::setObject(const std::string &member,
                                 const std::shared_ptr<Object> &value)
    {
      if (Sampler::setObject(member,value)) return true;

      if (member == "textureData") {
        textureData = value->as<TextureData>();
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

    TextureSampler::PLD *TextureSampler::getPLD(Device *device) 
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }
    
    void TextureSampler::commit() 
    {
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        if (pld->rtcTexture)
          device->rtc->freeTexture(pld->rtcTexture);
        pld->rtcTexture = 0;
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
        desc.borderColor    = borderColor;
        for (auto device : *devices) {
          PLD *pld = getPLD(device);
          if (pld->rtcTexture)
            device->rtc->freeTexture(pld->rtcTexture);
          pld->rtcTexture
            = textureData->getPLD(device)->rtc->createTexture(desc);
        };
      }

      Sampler::commit();
    }

    Sampler::DD TextureSampler::getDD(Device *device) 
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

      PLD *pld = getPLD(device);
      if (!pld->rtcTexture) {
        std::cout << "WARN: NO TEXTURE DATA ON IMAGE SAMPLER!" << std::endl;
        dd.texture = 0;
      } else {
        dd.texture = pld->rtcTexture->getDD();
        dd.numChannels = textureData->numChannels;
      }
      return dd;
    }
    
    Sampler::DD TransformSampler::getDD(Device *device) 
    {
      Sampler::DD dd;
      dd.type = Sampler::TRANSFORM;
      (vec4f&)dd.outTransform.offset = outOffset;
      memcpy(&dd.outTransform.mat_x,&outTransform,sizeof(outTransform));
      return dd;
    }

  }
}
