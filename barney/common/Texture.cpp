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

#include "barney/common/Texture.h"
#include "barney/Context.h"
#include "barney/ModelSlot.h"
#include "barney/DeviceGroup.h"
#include <barney.h>

namespace barney {

  rtc::Texture::ColorSpace toRTC(BNTextureColorSpace mode)
  { return 
      (mode == BN_COLOR_SPACE_LINEAR)
      ? rtc::Texture::COLOR_SPACE_LINEAR
      : rtc::Texture::COLOR_SPACE_SRGB;
  }
  
  rtc::Texture::FilterMode toRTC(BNTextureFilterMode mode)
  { return 
      (mode == BN_TEXTURE_NEAREST)
      ? rtc::Texture::FILTER_MODE_NEAREST
      : rtc::Texture::FILTER_MODE_LINEAR;
  }
  
  rtc::Texture::AddressMode toRTC(BNTextureAddressMode mode)
  {
    switch(mode) {
    case BN_TEXTURE_WRAP: return rtc::Texture::WRAP;
    case BN_TEXTURE_MIRROR: return rtc::Texture::MIRROR;
    case BN_TEXTURE_CLAMP: return rtc::Texture::CLAMP;
    case BN_TEXTURE_BORDER: return rtc::Texture::BORDER;
    default:
      BARNEY_INVALID_VALUE();
    }
  }
  
  Texture::Texture(Context *context, int slot,
                   BNDataType texelFormat,
                   vec2i size,
                   const void *texels,
                   BNTextureFilterMode  filterMode,
                   BNTextureAddressMode addressMode_x,
                   BNTextureAddressMode addressMode_y,
                   BNTextureColorSpace  colorSpace)
    : SlottedObject(context,slot)
  {
    rtc::TextureData::Format format;
    switch(texelFormat) {
    case BN_FLOAT:
      // owlTexelFormat = OWL_TEXEL_FORMAT_R32F;
      format = rtc::TextureData::FLOAT;
      break;
    case BN_UFIXED8:
      // owlTexelFormat = OWL_TEXEL_FORMAT_R8;
      format = rtc::TextureData::UCHAR;
      break;
    case BN_FLOAT4_RGBA:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
      format = rtc::TextureData::FLOAT4;
      break;
    case BN_FLOAT4:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
      format = rtc::TextureData::FLOAT4;
      break;
    case BN_UFIXED8_RGBA:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA8;
      format = rtc::TextureData::UCHAR4;
      break;
    default: throw std::runtime_error("un-recognized texel format "
                                      +std::to_string((int)texelFormat));
    }
    rtcTextureData
      = getRTC()->createTextureData(vec3i(size.x,size.y,0),
                                    format,texels);

    rtc::Texture::FilterMode rtcFilterMode
      = toRTC(filterMode);
    rtc::Texture::AddressMode rtcAddressModes[3] = {
      toRTC(addressMode_x),
      toRTC(addressMode_y),
      (rtc::Texture::AddressMode)0
    };
    rtc::Texture::ColorSpace rtcColorSpace
      = toRTC(colorSpace);
    
    rtcTexture
      = getRTC()->createTexture(rtcTextureData,
                                rtcFilterMode,
                                rtcAddressModes,
                                vec4f(0.f),
                                rtcColorSpace);
                                
    // assert(OWL_TEXTURE_NEAREST == (int)BN_TEXTURE_NEAREST);
    // assert(OWL_TEXTURE_LINEAR  == (int)BN_TEXTURE_LINEAR);
    
    // assert(OWL_TEXTURE_WRAP   == (int)BN_TEXTURE_WRAP);
    // assert(OWL_TEXTURE_CLAMP  == (int)BN_TEXTURE_CLAMP);
    // assert(OWL_TEXTURE_BORDER == (int)BN_TEXTURE_BORDER);
    // assert(OWL_TEXTURE_MIRROR == (int)BN_TEXTURE_MIRROR);

    // OWLTexelFormat owlTexelFormat;
    // switch(texelFormat) {
    // case BN_FLOAT:
    //   owlTexelFormat = OWL_TEXEL_FORMAT_R32F;
    //   break;
    // case BN_UFIXED8:
    //   owlTexelFormat = OWL_TEXEL_FORMAT_R8;
    //   break;
    // case BN_FLOAT4_RGBA:
    //   owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
    //   break;
    // case BN_FLOAT4:
    //   owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
    //   break;
    // case BN_UFIXED8_RGBA:
    //   owlTexelFormat = OWL_TEXEL_FORMAT_RGBA8;
    //   break;
    // default: throw std::runtime_error("un-recognized texel format "
    //                                   +std::to_string((int)texelFormat));
    // }
    
    // owlTexture 
    //   = owlTexture2DCreate(context->getOWL(slot),
    //                        owlTexelFormat,
    //                        size.x,size.y,
    //                        texels,
    //                        (OWLTextureFilterMode)filterMode,
    //                        (OWLTextureAddressMode)addressMode_x,
    //                        (OWLTextureAddressMode)addressMode_y,
    //                        // (OWLTextureColorSpace)colorSpace
    //                        OWL_COLOR_SPACE_LINEAR
    //                        );
  }
  
  Texture3D::Texture3D(Context *context, int slot,
                       BNDataType texelFormat,
                       vec3i size,
                       const void *texels,
                       BNTextureFilterMode  filterMode,
                       BNTextureAddressMode addressMode)
    : SlottedObject(context,slot)
  {
    rtc::TextureData::Format format;
    switch(texelFormat) {
    case BN_FLOAT:
      // owlTexelFormat = OWL_TEXEL_FORMAT_R32F;
      format = rtc::TextureData::FLOAT;
      break;
    case BN_UFIXED8:
      // owlTexelFormat = OWL_TEXEL_FORMAT_R8;
      format = rtc::TextureData::UCHAR;
      break;
    case BN_FLOAT4_RGBA:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
      format = rtc::TextureData::FLOAT4;
      break;
    case BN_FLOAT4:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
      format = rtc::TextureData::FLOAT4;
      break;
    case BN_UFIXED8_RGBA:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA8;
      format = rtc::TextureData::UCHAR4;
      break;
    default: throw std::runtime_error("un-recognized texel format "
                                      +std::to_string((int)texelFormat));
    }
    rtcTextureData
      = getRTC()->createTextureData(size,format,texels);

    rtc::Texture::FilterMode rtcFilterMode
      = toRTC(filterMode);
    rtc::Texture::AddressMode rtcAddressModes[3] = {
      toRTC(addressMode),
      toRTC(addressMode),
      toRTC(addressMode)
    };
    rtcTexture
      = getRTC()->createTexture(rtcTextureData,
                                rtcFilterMode,
                                rtcAddressModes);
    rtcTextureNN
      = getRTC()->createTexture(rtcTextureData,
                                rtc::Texture::FILTER_MODE_NEAREST,
                                rtcAddressModes);


    // if (!tex3Ds.empty()) return;

    // auto &devices = getDevices();
    // tex3Ds.resize(context->contextSize());

    // cudaChannelFormatDesc desc;
    // cudaTextureReadMode   readMode;
    // size_t sizeOfScalar;
    // int numScalarsPerTexel;
    // switch (texelFormat) {
    // case BN_FLOAT:
    //   desc         = cudaCreateChannelDesc<float>();
    //   sizeOfScalar = 4;
    //   readMode     = cudaReadModeElementType;
    //   numScalarsPerTexel = 1;
    //   break;
    // case BN_UFIXED8:
    //   desc         = cudaCreateChannelDesc<uint8_t>();
    //   sizeOfScalar = 1;
    //   readMode     = cudaReadModeNormalizedFloat;
    //   numScalarsPerTexel = 1;
    //   break;
    // case BN_UFIXED8_RGBA:
    //   desc         = cudaCreateChannelDesc<uchar4>();
    //   sizeOfScalar = 1;
    //   readMode     = cudaReadModeNormalizedFloat;
    //   numScalarsPerTexel = 4;
    //   break;
    // case BN_UFIXED16:
    //   desc         = cudaCreateChannelDesc<uint8_t>();
    //   sizeOfScalar = 2;
    //   readMode     = cudaReadModeNormalizedFloat;
    //   numScalarsPerTexel = 1;
    //   break;
    // default:
    //   throw std::runtime_error("Texture3D with non-implemented scalar type ...");
    // }
    // // for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
    //   // auto dev = devGroup->devices[lDevID];
    // for (auto dev : devices) {
    //   auto &tex = getDD(dev);//tex3Ds[dev->contextRank];
    //   SetActiveGPU forDuration(dev);
    //   // Copy voxels to cuda array
    //   cudaExtent extent{
    //     (unsigned)size.x,
    //     (unsigned)size.y,
    //     (unsigned)size.z};
    //   BARNEY_CUDA_CALL(Malloc3DArray(&tex.voxelArray,&desc,extent,0));
    //   cudaMemcpy3DParms copyParms;
    //   memset(&copyParms,0,sizeof(copyParms));
    //   copyParms.srcPtr
    //     = make_cudaPitchedPtr((void *)texels,
    //                           (size_t)size.x*sizeOfScalar*numScalarsPerTexel,
    //                           (size_t)size.x,
    //                           (size_t)size.y);
    //   copyParms.dstArray = tex.voxelArray;
    //   copyParms.extent   = extent;
    //   copyParms.kind     = cudaMemcpyHostToDevice;
    //   BARNEY_CUDA_CALL(Memcpy3D(&copyParms));
          
    //   // Create a texture object
    //   cudaResourceDesc resourceDesc;
    //   memset(&resourceDesc,0,sizeof(resourceDesc));
    //   resourceDesc.resType         = cudaResourceTypeArray;
    //   resourceDesc.res.array.array = tex.voxelArray;
          
    //   cudaTextureDesc textureDesc;
    //   memset(&textureDesc,0,sizeof(textureDesc));
    //   textureDesc.addressMode[0]   = cudaAddressModeClamp;
    //   textureDesc.addressMode[1]   = cudaAddressModeClamp;
    //   textureDesc.addressMode[2]   = cudaAddressModeClamp;
    //   textureDesc.filterMode       = cudaFilterModeLinear;
    //   textureDesc.readMode         = readMode;
    //     // = (texelFormat == BN_TEXEL_FORMAT_R8)
    //     // ? cudaReadModeNormalizedFloat
    //     // : cudaReadModeElementType;
    //   // textureDesc.readMode         = cudaReadModeElementType;
    //   textureDesc.normalizedCoords = false;
          
    //   BARNEY_CUDA_CALL(CreateTextureObject(&tex.texObj,
    //                                        &resourceDesc,
    //                                        &textureDesc,0));
          
    //   // 2nd texture object for nearest filtering
    //   textureDesc.filterMode       = cudaFilterModePoint;
    //   BARNEY_CUDA_CALL(CreateTextureObject(&tex.texObjNN,
    //                                        &resourceDesc,
    //                                        &textureDesc,0));
    // }
  }

  // Texture3D::DD &Texture3D::getDD(const std::shared_ptr<Device> &device) 
  // {
  //   return tex3Ds[device->contextRank];
  // }


  Texture3D::DD Texture3D::getDD(const std::shared_ptr<Device> &device)
  {
    return getDD(device->rtc);
  }
  
  Texture3D::DD Texture3D::getDD(const rtc::Device *device)
  {
    Texture3D::DD dd;
    dd.texObj = rtcTexture->getDD(device);
    dd.texObjNN = rtcTextureNN->getDD(device);
    return dd;
  }
  
  

  TextureData::TextureData(Context *context, int slot,
                           BNDataType texelFormat,
                           vec3i size,
                           const void *texels)
    : SlottedObject(context,slot),
      dims(size),
      texelFormat(texelFormat)
  {
    rtc::TextureData::Format format;
    switch(texelFormat) {
    case BN_FLOAT:
      // owlTexelFormat = OWL_TEXEL_FORMAT_R32F;
      format = rtc::TextureData::FLOAT;
      break;
    case BN_UFIXED8:
      // owlTexelFormat = OWL_TEXEL_FORMAT_R8;
      format = rtc::TextureData::UCHAR;
      break;
    case BN_FLOAT4_RGBA:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
      format = rtc::TextureData::FLOAT4;
      break;
    case BN_FLOAT4:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA32F;
      format = rtc::TextureData::FLOAT4;
      break;
    case BN_UFIXED8_RGBA:
      // owlTexelFormat = OWL_TEXEL_FORMAT_RGBA8;
      format = rtc::TextureData::UCHAR4;
      break;
    default: throw std::runtime_error("un-recognized texel format "
                                      +std::to_string((int)texelFormat));
    }
    rtcTextureData
      = getRTC()->createTextureData(vec3i(size.x,size.y,0),
                                    format,texels);
    // if (!onDev.empty()) return;

    // rtc::TextureData::Format format;
    // switch (texelFormat) {
    // case BN_FLOAT:
    //   // desc         = cudaCreateChannelDesc<float>();
    //   // sizeOfScalar = 4;
    //   // // readMode     = cudaReadModeElementType;
    //   // numScalarsPerTexel = 1;
    //   format = rtc::TextureData::FLOAT;
    //   break;
    // case BN_FLOAT4:
    //   // desc         = cudaCreateChannelDesc<float4>();
    //   // sizeOfScalar = 4;
    //   // // readMode     = cudaReadModeElementType;
    //   // numScalarsPerTexel = 4;
    //   format = rtc::TextureData::FLOAT4;
    //   break;
    // case BN_UFIXED8:
    //   // desc         = cudaCreateChannelDesc<uint8_t>();
    //   // sizeOfScalar = 1;
    //   // // readMode     = cudaReadModeNormalizedFloat;
    //   // numScalarsPerTexel = 1;
    //   format = rtc::TextureData::UCHAR;
    //   break;
    // case BN_UFIXED8_RGBA:
    //   // desc         = cudaCreateChannelDesc<uchar4>();
    //   // sizeOfScalar = 1;
    //   // // readMode     = cudaReadModeNormalizedFloat;
    //   // numScalarsPerTexel = 4;
    //   format = rtc::TextureData::UCHAR4;
    //   break;
    // case BN_UFIXED16:
    //   // desc         = cudaCreateChannelDesc<uint16_t>();
    //   // sizeOfScalar = 2;
    //   // // readMode     = cudaReadModeNormalizedFloat;
    //   // numScalarsPerTexel = 1;
    //   format = rtc::TextureData::USHORT;
    //   break;
    // default:
    //   throw std::runtime_error("TextureData with non-implemented scalar type ...");
    
    // rtcTextureData
    //   = getRTC()->createTextureData(size,format,texels);
      
    // auto &devices = getDevices();
    // onDev.resize(context->contextSize());

    // cudaChannelFormatDesc desc;
    // // cudaTextureReadMode   readMode;
    // size_t sizeOfScalar;
    // int    numScalarsPerTexel;
    // switch (texelFormat) {
    // case BN_FLOAT:
    //   desc         = cudaCreateChannelDesc<float>();
    //   sizeOfScalar = 4;
    //   // readMode     = cudaReadModeElementType;
    //   numScalarsPerTexel = 1;
    //   break;
    // case BN_FLOAT4:
    //   desc         = cudaCreateChannelDesc<float4>();
    //   sizeOfScalar = 4;
    //   // readMode     = cudaReadModeElementType;
    //   numScalarsPerTexel = 4;
    //   break;
    // case BN_UFIXED8:
    //   desc         = cudaCreateChannelDesc<uint8_t>();
    //   sizeOfScalar = 1;
    //   // readMode     = cudaReadModeNormalizedFloat;
    //   numScalarsPerTexel = 1;
    //   break;
    // case BN_UFIXED8_RGBA:
    //   desc         = cudaCreateChannelDesc<uchar4>();
    //   sizeOfScalar = 1;
    //   // readMode     = cudaReadModeNormalizedFloat;
    //   numScalarsPerTexel = 4;
    //   break;
    // case BN_UFIXED16:
    //   desc         = cudaCreateChannelDesc<uint16_t>();
    //   sizeOfScalar = 2;
    //   // readMode     = cudaReadModeNormalizedFloat;
    //   numScalarsPerTexel = 1;
    //   break;
    // default:
    //   throw std::runtime_error("TextureData with non-implemented scalar type ...");
    // }

    // for (auto dev : devices) {
    //   auto &dd = getDD(dev);//onDev[dev->localRank];
    //   SetActiveGPU forDuration(dev);
    //   BARNEY_CUDA_CALL(MallocArray(&dd.array,&desc,size.x,size.y,0));
    //   BARNEY_CUDA_CALL(Memcpy2DToArray(dd.array,0,0,
    //                                    (void *)texels,
    //                                    (size_t)size.x*sizeOfScalar*numScalarsPerTexel,
    //                                    (size_t)size.x*sizeOfScalar*numScalarsPerTexel,
    //                                    (size_t)size.y,
    //                                    cudaMemcpyHostToDevice));
    // }
  }

  // TextureData::DD &TextureData::getDD(const std::shared_ptr<Device> &device)
  // {
  //   return onDev[device->contextRank];
  // }
  

  TextureData::~TextureData()
  {
    // auto devGroup = owner->devGroup.get();
    // for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
    getRTC()->free(rtcTextureData);
    rtcTextureData = 0;
    // for (auto dev : getDevices()) {
    //   auto &dd = getDD(dev);//onDev[dev->localRank];
    //   SetActiveGPU forDuration(dev);
    //   if (dd.array)
    //     BARNEY_CUDA_CALL_NOTHROW(FreeArray(dd.array));
    //   dd.array = 0;
    // }
  }

  // cudaTextureObject_t
  rtc::device::TextureObject
  Texture::getTextureObject(const Device *device) const
  {
    // return (cudaTextureObject_t)owlTextureGetObject(owlTexture,device->owlID);
    return rtcTexture->getDD(device->rtc);
  }
    
  
}
