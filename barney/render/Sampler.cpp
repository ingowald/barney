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
#include "barney/common/Texture.h"
#include "barney/ModelSlot.h"

namespace barney {
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
      if (attributeName == "color")
        return render::COLOR; 
      
      throw std::runtime_error("PossiblyMappedParameter::set not implemented for attribute '"+attributeName+"'");
    }

    Sampler::Sampler(ModelSlot *owner)
      : SlottedObject(owner),
        samplerID(owner->world.samplerLibrary->allocate())
    {
      perDev.resize(owner->devGroup->size());
    }
    
    Sampler::~Sampler()
    {
      owner->world.samplerLibrary->release(samplerID);
      for (int devID=0;devID<owner->devGroup->size();devID++) {
        auto dev = owner->devGroup->devices[devID];
        SetActiveGPU forDuration(dev);

        Sampler::DD &dd = perDev[devID];
        freeDD(dd,devID);
        /*! MIGHT want to upload something to device sampler library
            that indicates that this is now dead - but if it is, and
            somebody else were to stell use it, then that's a bug on
            that somebody else's side, anyway (OR it is a bug on the user's side */
      }
    }
    
    Sampler::SP Sampler::create(ModelSlot *owner, const std::string &type)
    {
      if (type == "texture1D")
        return std::make_shared<TextureSampler>(owner,1);
      if (type == "texture2D" || type == "image2D")
        return std::make_shared<TextureSampler>(owner,2);
      if (type == "texture3D")
        return std::make_shared<TextureSampler>(owner,3);
      if (type == "transform")
        return std::make_shared<TransformSampler>(owner);
      throw std::runtime_error("do not know what a '"+type+" sampler is !?");
    }


    bool Sampler::setObject(const std::string &member,
                                 const std::shared_ptr<Object> &value)
    {
      if (SlottedObject::setObject(member,value)) return true;

      return false;
    }
        
    bool Sampler::setString(const std::string &member,
                            const std::string &value)
    {
      if (SlottedObject::setString(member,value)) return true;

      if (member == "inAttribute")
        { inAttribute = parseAttribute(value); return true; }
      
      return false;
    }

    bool Sampler::set4x4f(const std::string &member, const mat4f &value)
    {
      if (SlottedObject::set4x4f(member,value)) return true;

      if (member == "outTransform")
        { outTransform = value; return true; }
      
      return false;
    }
    
    bool Sampler::set4f(const std::string &member, const vec4f &value)
    {
      if (SlottedObject::set4f(member,value)) return true;

      if (member == "outOffset")
        { outOffset = value; return true; }

      return false;
    }

    void Sampler::commit() 
    {
      SlottedObject::commit();

      for (int devID=0;devID<owner->devGroup->size();devID++) {
        auto dev = owner->devGroup->devices[devID];
        SetActiveGPU forDuration(dev);

        Sampler::DD &dd = perDev[devID];
        freeDD(dd,devID);
        createDD(dd,devID);
        owner->world.samplerLibrary->setDD(samplerID,dd,devID);
      }
    }


    bool TextureSampler::setObject(const std::string &member,
                                 const std::shared_ptr<Object> &value)
    {
      if (Sampler::setObject(member,value)) return true;

      if (member == "textureData")
        { textureData = value->as<TextureData>(); return true; }

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
    
    bool TextureSampler::set4x4f(const std::string &member, const mat4f &value)
    {
      if (Sampler::set4x4f(member,value)) return true;

      if (member == "inTransform")
        { inTransform = value; return true; }      

      return false;
    }
    
    bool TextureSampler::set4f(const std::string &member, const vec4f &value)
    {
      if (Sampler::set4f(member,value)) return true;

      if (member == "inOffset")
        { inOffset = value; return true; }
      
      return false;
    }

    void TextureSampler::commit() 
    {
      Sampler::commit();
    }
    
    void TextureSampler::createDD(DD &dd, int devID)
    {
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
      memcpy(&dd.outTransform.mat,&outTransform,sizeof(outTransform));
      
      (vec4f&)dd.image.inTransform.offset = inOffset;
      memcpy(&dd.image.inTransform.mat,&inTransform,sizeof(inTransform));
      
      dd.image.numChannels = numDims;
      
      if (!textureData) {
        std::cout << "WARN: NO TEXTURE DATA ON IMAGE SAMPLER!" << std::endl;
        dd.image.texture = 0;
        return;
      }

      // Create a texture object
      cudaResourceDesc resourceDesc;
      memset(&resourceDesc,0,sizeof(resourceDesc));
      resourceDesc.resType         = cudaResourceTypeArray;
      resourceDesc.res.array.array = textureData->onDev[devID].array;
      
      cudaTextureDesc tex_desc;
      memset(&tex_desc,0,sizeof(tex_desc));
      for (int i=0;i<3;i++)
        switch (wrapModes[i]) {
        case BN_TEXTURE_WRAP:
          tex_desc.addressMode[i] = cudaAddressModeWrap;
          break;
        case BN_TEXTURE_CLAMP:
          tex_desc.addressMode[i] = cudaAddressModeClamp;
          break;
        case BN_TEXTURE_BORDER:
          tex_desc.addressMode[i] = cudaAddressModeBorder;
          break;
        case BN_TEXTURE_MIRROR:
          tex_desc.addressMode[i] = cudaAddressModeMirror;
          break;
        default:
          throw std::runtime_error("invalid texture mode!?");
        }
      tex_desc.filterMode       =
        (filterMode == BN_TEXTURE_LINEAR)
        ? cudaFilterModeLinear
        : cudaFilterModePoint;
      tex_desc.normalizedCoords = 1;
      tex_desc.maxAnisotropy       = 1;
      tex_desc.maxMipmapLevelClamp = 0;
      // tex_desc.maxMipmapLevelClamp = 99;
      tex_desc.minMipmapLevelClamp = 0;
      tex_desc.mipmapFilterMode    = cudaFilterModePoint;
      tex_desc.borderColor[0]      = 1.0f;
      tex_desc.borderColor[1]      = 0.0f;
      tex_desc.borderColor[2]      = 0.0f;
      tex_desc.borderColor[3]      = 1.0f;
      tex_desc.sRGB                = 0;//1;//(colorSpace == OWL_COLOR_SPACE_SRGB);

      switch (textureData->texelFormat) {
      case BN_TEXEL_FORMAT_R32F:
        tex_desc.readMode     = cudaReadModeElementType;
        break;
      case BN_TEXEL_FORMAT_R8:
        tex_desc.readMode     = cudaReadModeNormalizedFloat;
        break;
      case BN_TEXEL_FORMAT_RGBA8:
        tex_desc.readMode     = cudaReadModeNormalizedFloat;
        break;
      case BN_TEXEL_FORMAT_R16:
        tex_desc.readMode     = cudaReadModeNormalizedFloat;
        break;
      default:
        throw std::runtime_error("unsupported texel format in image sampler");
      }
          
      BARNEY_CUDA_CALL(CreateTextureObject(&dd.image.texture,&resourceDesc,&tex_desc,0));
    }
    
    void TextureSampler::freeDD(DD &dd, int devID)
    {
      if (dd.image.texture)
        BARNEY_CUDA_CALL(DestroyTextureObject(dd.image.texture));
      dd.image.texture = 0;
      dd.type = Sampler::INVALID;
    }
    
    void TransformSampler::createDD(DD &dd, int devID)
    {
      dd.type = Sampler::TRANSFORM;
      (vec4f&)dd.outTransform.offset = outOffset;
      memcpy(&dd.outTransform.mat,&outTransform,sizeof(outTransform));
    }

    void TransformSampler::freeDD(DD &dd, int devID)
    {
      /* no device data to free for this one ... */
      dd.type = Sampler::INVALID;
      
    }

  }
}
