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

namespace barney {

  Texture::Texture(ModelSlot *owner,
                   BNTexelFormat texelFormat,
                   vec2i size,
                   const void *texels,
                   BNTextureFilterMode  filterMode,
                   BNTextureAddressMode addressMode,
                   BNTextureColorSpace  colorSpace)
    : Object(owner->context)
  {
    assert(OWL_TEXEL_FORMAT_RGBA8   == (int)BN_TEXEL_FORMAT_RGBA8);
    assert(OWL_TEXEL_FORMAT_RGBA32F == (int)BN_TEXEL_FORMAT_RGBA32F);
    
    assert(OWL_TEXTURE_NEAREST == (int)BN_TEXTURE_NEAREST);
    assert(OWL_TEXTURE_LINEAR  == (int)BN_TEXTURE_LINEAR);
    
    assert(OWL_TEXTURE_WRAP   == (int)BN_TEXTURE_WRAP);
    assert(OWL_TEXTURE_CLAMP  == (int)BN_TEXTURE_CLAMP);
    assert(OWL_TEXTURE_BORDER == (int)BN_TEXTURE_BORDER);
    assert(OWL_TEXTURE_MIRROR == (int)BN_TEXTURE_MIRROR);
    
    owlTex = owlTexture2DCreate(owner->getOWL(),
                                (OWLTexelFormat)texelFormat,
                                size.x,size.y,
                                texels,
                                (OWLTextureFilterMode)filterMode,
                                (OWLTextureAddressMode)addressMode,
                                // (OWLTextureColorSpace)colorSpace
                                OWL_COLOR_SPACE_LINEAR
                                );
  }

  Texture3D::Texture3D(ModelSlot *owner,
                       BNTexelFormat texelFormat,
                       vec3i size,
                       const void *texels,
                       BNTextureFilterMode  filterMode,
                       BNTextureAddressMode addressMode)
    : SlottedObject(owner)
  {
    if (!tex3Ds.empty()) return;

    auto devGroup = owner->devGroup.get();
    tex3Ds.resize(devGroup->size());

    cudaChannelFormatDesc desc;
    cudaTextureReadMode   readMode;
    size_t sizeOfScalar;
    int numScalarsPerTexel;
    switch (texelFormat) {
    case BN_TEXEL_FORMAT_R32F:
      desc         = cudaCreateChannelDesc<float>();
      sizeOfScalar = 4;
      readMode     = cudaReadModeElementType;
      numScalarsPerTexel = 1;
      break;
    case BN_TEXEL_FORMAT_R8:
      desc         = cudaCreateChannelDesc<uint8_t>();
      sizeOfScalar = 1;
      readMode     = cudaReadModeNormalizedFloat;
      numScalarsPerTexel = 1;
      break;
    case BN_TEXEL_FORMAT_RGBA8:
      desc         = cudaCreateChannelDesc<uchar4>();
      sizeOfScalar = 1;
      readMode     = cudaReadModeNormalizedFloat;
      numScalarsPerTexel = 4;
      break;
    case BN_TEXEL_FORMAT_R16:
      desc         = cudaCreateChannelDesc<uint8_t>();
      sizeOfScalar = 2;
      readMode     = cudaReadModeNormalizedFloat;
      numScalarsPerTexel = 1;
      break;
    default:
      throw std::runtime_error("Texture3D with non-implemented scalar type ...");
    }
    // if (scalarType != BN_FLOAT)
    //   throw std::runtime_error("can only do float 3d texs..");

    std::cout << "#bn.struct: creating CUDA 3D textures" << std::endl;
    for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
      auto dev = devGroup->devices[lDevID];
      auto &tex = tex3Ds[lDevID];
      SetActiveGPU forDuration(dev);
      // Copy voxels to cuda array
      cudaExtent extent{
        (unsigned)size.x,
        (unsigned)size.y,
        (unsigned)size.z};
      BARNEY_CUDA_CALL(Malloc3DArray(&tex.voxelArray,&desc,extent,0));
      cudaMemcpy3DParms copyParms;
      memset(&copyParms,0,sizeof(copyParms));
      copyParms.srcPtr = make_cudaPitchedPtr((void *)texels,
                                             (size_t)size.x*sizeOfScalar*numScalarsPerTexel,
                                             (size_t)size.x,
                                             (size_t)size.y);
      copyParms.dstArray = tex.voxelArray;
      copyParms.extent   = extent;
      copyParms.kind     = cudaMemcpyHostToDevice;
      BARNEY_CUDA_CALL(Memcpy3D(&copyParms));
          
      // Create a texture object
      cudaResourceDesc resourceDesc;
      memset(&resourceDesc,0,sizeof(resourceDesc));
      resourceDesc.resType         = cudaResourceTypeArray;
      resourceDesc.res.array.array = tex.voxelArray;
          
      cudaTextureDesc textureDesc;
      memset(&textureDesc,0,sizeof(textureDesc));
      textureDesc.addressMode[0]   = cudaAddressModeClamp;
      textureDesc.addressMode[1]   = cudaAddressModeClamp;
      textureDesc.addressMode[2]   = cudaAddressModeClamp;
      textureDesc.filterMode       = cudaFilterModeLinear;
      textureDesc.readMode         = readMode;
        // = (texelFormat == BN_TEXEL_FORMAT_R8)
        // ? cudaReadModeNormalizedFloat
        // : cudaReadModeElementType;
      // textureDesc.readMode         = cudaReadModeElementType;
      textureDesc.normalizedCoords = false;
          
      BARNEY_CUDA_CALL(CreateTextureObject(&tex.texObj,&resourceDesc,&textureDesc,0));
          
      // 2nd texture object for nearest filtering
      textureDesc.filterMode       = cudaFilterModePoint;
      BARNEY_CUDA_CALL(CreateTextureObject(&tex.texObjNN,&resourceDesc,&textureDesc,0));
    }
  }


  TextureData::TextureData(ModelSlot *owner,
                           BNTexelFormat texelFormat,
                           vec3i size,
                           const void *texels)
    : SlottedObject(owner),
      dims(size),
      texelFormat(texelFormat)
  {
    if (!onDev.empty()) return;

    auto devGroup = owner->devGroup.get();
    onDev.resize(devGroup->size());

    cudaChannelFormatDesc desc;
    // cudaTextureReadMode   readMode;
    size_t sizeOfScalar;
    int    numScalarsPerTexel;
    switch (texelFormat) {
    case BN_TEXEL_FORMAT_R32F:
      desc         = cudaCreateChannelDesc<float>();
      sizeOfScalar = 4;
      // readMode     = cudaReadModeElementType;
      numScalarsPerTexel = 1;
      break;
    case BN_TEXEL_FORMAT_R8:
      desc         = cudaCreateChannelDesc<uint8_t>();
      sizeOfScalar = 1;
      // readMode     = cudaReadModeNormalizedFloat;
      numScalarsPerTexel = 1;
      break;
    case BN_TEXEL_FORMAT_RGBA8:
      desc         = cudaCreateChannelDesc<uchar4>();
      sizeOfScalar = 1;
      // readMode     = cudaReadModeNormalizedFloat;
      numScalarsPerTexel = 4;
      break;
    case BN_TEXEL_FORMAT_R16:
      desc         = cudaCreateChannelDesc<uint16_t>();
      sizeOfScalar = 2;
      // readMode     = cudaReadModeNormalizedFloat;
      numScalarsPerTexel = 1;
      break;
    default:
      throw std::runtime_error("TextureData with non-implemented scalar type ...");
    }

    for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
      auto dev = devGroup->devices[lDevID];
      auto &dd = onDev[lDevID];
      SetActiveGPU forDuration(dev);
      BARNEY_CUDA_CALL(MallocArray(&dd.array,&desc,size.x,size.y,0));
      BARNEY_CUDA_CALL(Memcpy2DToArray(dd.array,0,0,
                                       (void *)texels,
                                       (size_t)size.x*sizeOfScalar*numScalarsPerTexel,
                                       (size_t)size.x*sizeOfScalar*numScalarsPerTexel,
                                       (size_t)size.y,
                                       cudaMemcpyHostToDevice));
    }
  }

  TextureData::~TextureData()
  {
    auto devGroup = owner->devGroup.get();
    for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
      auto dev = devGroup->devices[lDevID];
      auto &dd = onDev[lDevID];
      SetActiveGPU forDuration(dev);
      if (dd.array)
        BARNEY_CUDA_CALL_NOTHROW(FreeArray(dd.array));
      dd.array = 0;
    }
  }
  
}
