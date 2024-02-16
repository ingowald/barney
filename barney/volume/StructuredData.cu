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

#include "barney/volume/StructuredData.h"
#include "barney/Context.h"

namespace barney {

  extern "C" char StructuredData_ptx[];

  enum { cellsPerMC = 8 };

  __global__
  void computeMCs(MCGrid::DD mcGrid,
                  vec3i numScalars,
                  cudaTextureObject_t texNN)
  {
    vec3i mcID = vec3i(threadIdx) + vec3i(blockIdx) * vec3i(blockDim);
    if (mcID.x >= mcGrid.dims.x) return;
    if (mcID.y >= mcGrid.dims.y) return;
    if (mcID.z >= mcGrid.dims.z) return;

    range1f scalarRange;
    for (int iiz=0;iiz<=cellsPerMC;iiz++)
      for (int iiy=0;iiy<=cellsPerMC;iiy++)
        for (int iix=0;iix<=cellsPerMC;iix++) {
          vec3i scalarID = mcID*int(cellsPerMC) + vec3i(iix,iiy,iiz);
          if (scalarID.x >= numScalars.x) continue;
          if (scalarID.y >= numScalars.y) continue;
          if (scalarID.z >= numScalars.z) continue;
          scalarRange.extend(tex3D<float>(texNN,scalarID.x,scalarID.y,scalarID.z));
        }
    int mcIdx = mcID.x + mcGrid.dims.x*(mcID.y+mcGrid.dims.y*(mcID.z));
    mcGrid.scalarRanges[mcIdx] = scalarRange;
    if (mcIdx == 3)
      printf("sanity: macro cell #3 has range %f %f\n",
             scalarRange.lower,scalarRange.upper);
  }
  
  // StructuredData::StructuredData(DataGroup *owner,
  //                                const vec3i &numScalars,
  //                                BNScalarType scalarType,
  //                                const void *scalars,
  //                                const vec3f &gridOrigin,
  //                                const vec3f &gridSpacing)
  //   : ScalarField(owner),
  //     numScalars(numScalars),
  //     numCells(numScalars - 1),
  //     scalarType(scalarType),
  //     // rawScalarData(scalars),
  //     gridOrigin(gridOrigin),
  //     gridSpacing(gridSpacing)
  // {
  //   worldBounds.lower = gridOrigin;
  //   worldBounds.upper = gridOrigin + gridSpacing * vec3f(numScalars);
  //   createCUDATextures();
  // }


  StructuredData::StructuredData(DataGroup *owner)
    : ScalarField(owner)
  {}



  
  // void StructuredData::createCUDATextures()
  // {
  //   if (!tex3Ds.empty()) return;

  //   tex3Ds.resize(devGroup->size());

  //   cudaChannelFormatDesc desc;
  //   size_t sizeOfScalar;
  //   switch (scalarType) {
  //   case BN_SCALAR_FLOAT:
  //     desc = cudaCreateChannelDesc<float>();
  //     sizeOfScalar = 4;
  //     break;
  //   case BN_SCALAR_UINT8:
  //     desc = cudaCreateChannelDesc<uint8_t>();
  //     sizeOfScalar = 1;
  //     break;
  //   default:
  //     throw std::runtime_error("structured data with non-implemented scalar type ...");
  //   }
  //   // if (scalarType != BN_FLOAT)
  //   //   throw std::runtime_error("can only do float 3d texs..");

  //   std::cout << "#bn.struct: creating CUDA 3D textures" << std::endl;
  //   for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
  //     auto dev = devGroup->devices[lDevID];
  //     auto &tex = tex3Ds[lDevID];
  //     SetActiveGPU forDuration(dev);
  //     // Copy voxels to cuda array
  //     cudaExtent extent{
  //       (unsigned)numScalars.x,
  //       (unsigned)numScalars.y,
  //       (unsigned)numScalars.z};
  //     BARNEY_CUDA_CALL(Malloc3DArray(&tex.voxelArray,&desc,extent,0));
  //     cudaMemcpy3DParms copyParms;
  //     memset(&copyParms,0,sizeof(copyParms));
  //     copyParms.srcPtr = make_cudaPitchedPtr((void *)rawScalarData,
  //                                            (size_t)numScalars.x*sizeOfScalar,
  //                                            (size_t)numScalars.x,
  //                                            (size_t)numScalars.y);
  //     copyParms.dstArray = tex.voxelArray;
  //     copyParms.extent   = extent;
  //     copyParms.kind     = cudaMemcpyHostToDevice;
  //     BARNEY_CUDA_CALL(Memcpy3D(&copyParms));
          
  //     // Create a texture object
  //     cudaResourceDesc resourceDesc;
  //     memset(&resourceDesc,0,sizeof(resourceDesc));
  //     resourceDesc.resType         = cudaResourceTypeArray;
  //     resourceDesc.res.array.array = tex.voxelArray;
          
  //     cudaTextureDesc textureDesc;
  //     memset(&textureDesc,0,sizeof(textureDesc));
  //     textureDesc.addressMode[0]   = cudaAddressModeClamp;
  //     textureDesc.addressMode[1]   = cudaAddressModeClamp;
  //     textureDesc.addressMode[2]   = cudaAddressModeClamp;
  //     textureDesc.filterMode       = cudaFilterModeLinear;
  //     textureDesc.readMode
  //       = scalarType == BN_SCALAR_UINT8
  //       ? cudaReadModeNormalizedFloat
  //       : cudaReadModeElementType;
  //     // textureDesc.readMode         = cudaReadModeElementType;
  //     textureDesc.normalizedCoords = false;
          
  //     BARNEY_CUDA_CALL(CreateTextureObject(&tex.texObj,&resourceDesc,&textureDesc,0));
          
  //     // 2nd texture object for nearest filtering
  //     textureDesc.filterMode       = cudaFilterModePoint;
  //     BARNEY_CUDA_CALL(CreateTextureObject(&tex.texObjNN,&resourceDesc,&textureDesc,0));
  //   }
  //   PING;
  // }
  
  

  void StructuredData::buildMCs(MCGrid &mcGrid) 
  {
    vec3i mcDims = divRoundUp(numCells,vec3i(cellsPerMC));
    mcGrid.resize(mcDims);
    vec3i blockSize(4);
    vec3i numBlocks = divRoundUp(mcDims,blockSize);
    mcGrid.gridOrigin = worldBounds.lower;
    mcGrid.gridSpacing = vec3f(cellsPerMC) * this->gridSpacing;
    std::cout << "building macro cells for grid of " << mcDims << " macro cells" << std::endl;
    for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
      auto dev = devGroup->devices[lDevID];
      SetActiveGPU forDuration(dev);
      computeMCs<<<(const dim3&)numBlocks,(const dim3&)blockSize>>>
        (mcGrid.getDD(lDevID),numScalars,
         texture->tex3Ds[lDevID].texObjNN);
    }
    BARNEY_CUDA_SYNC_CHECK();
  }
  
  // ScalarField *DataGroup::createStructuredData(const vec3i &numScalars,
  //                                              BNScalarType scalarType,
  //                                              const void *data,
  //                                              const vec3f &gridOrigin,
  //                                              const vec3f &gridSpacing)
  // {
  //   StructuredData::SP sf
  //     = std::make_shared<StructuredData>(this,
  //                                        numScalars,scalarType,data,
  //                                        gridOrigin,gridSpacing);
  //   return getContext()->initReference(sf);
  // }

  void StructuredData::setVariables(OWLGeom geom)
  {
    ScalarField::setVariables(geom);
    
    for (int lDevID=0;lDevID<devGroup->devices.size();lDevID++) {
      cudaTextureObject_t tex = texture->tex3Ds[lDevID].texObj;
      owlGeomSetRaw(geom,"tex3D",&tex,lDevID);
    }
    owlGeomSet3f(geom,"cellGridOrigin",
                 gridOrigin.x,
                 gridOrigin.y,
                 gridOrigin.z);
    owlGeomSet3f(geom,"cellGridSpacing",
                 gridSpacing.x,
                 gridSpacing.y,
                 gridSpacing.z);
    owlGeomSet3i(geom,"numCells",
                 numCells.x,
                 numCells.y,
                 numCells.z);
  }
  
  VolumeAccel::SP StructuredData::createAccel(Volume *volume) 
  {
    const char *methodFromEnv = getenv("BARNEY_STRUCTURED");
    const std::string method = methodFromEnv ? methodFromEnv : "";
    if (method != "DDA")
      return std::make_shared<MCRTXVolumeAccel<StructuredDataSampler>::Host>
        (this,volume,StructuredData_ptx);
    else
      return std::make_shared<MCDDAVolumeAccel<StructuredDataSampler>::Host>
        (this,volume,StructuredData_ptx);
  }
  
  void StructuredData::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    ScalarField::DD::addVars(vars,base);
    vars.push_back
      ({"tex3D",OWL_USER_TYPE(cudaTextureObject_t),base+OWL_OFFSETOF(DD,texObj)});
    vars.push_back
      ({"cellGridOrigin",OWL_FLOAT3,base+OWL_OFFSETOF(DD,cellGridOrigin)});
    vars.push_back
      ({"cellGridSpacing",OWL_FLOAT3,base+OWL_OFFSETOF(DD,cellGridSpacing)});
    vars.push_back
      ({"numCells",OWL_INT3,base+OWL_OFFSETOF(DD,numCells)});
  }

  // ==================================================================
  bool StructuredData::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "gridOrigin") {
      gridOrigin = value;
      return true;
    }
    if (member == "gridSpacing") {
      gridSpacing = value;
      return true;
    }
    return false;
  }

  // ==================================================================
  bool StructuredData::set3i(const std::string &member, const vec3i &value) 
  {
    if (member == "dims") {
      numScalars = value;
      numCells   = value - vec3i(1);
      return true;
    }
    return false;
  }

  // ==================================================================
  bool StructuredData::setObject(const std::string &member, const Object::SP &value) 
  {
    if (member == "texture") {
      texture = value->as<Texture3D>();
      return true;
    }
    return false;
  }

  // ==================================================================
  void StructuredData::commit() 
  {
    worldBounds.lower = gridOrigin;
    worldBounds.upper = gridOrigin + gridSpacing * vec3f(numScalars);
  }

  
  
}

