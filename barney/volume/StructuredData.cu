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

#include "barney/Context.h"
#include "barney/volume/StructuredData.h"
#include "barney/common/Texture.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {

  RTC_IMPORT_USER_GEOM(/*file*/StructuredData,/*name*/StructuredMC,
                       /*geomtype device data */
                       MCVolumeAccel<StructuredDataSampler>::DD,false,false);
  RTC_IMPORT_COMPUTE3D(StructuredData_computeMCs);

  StructuredData::PLD *StructuredData::getPLD(Device *device) 
  { return &perLogical[device->contextRank()]; } 

  /*! how many cells (in each dimension) will go into a macro
      cell. eg, a value of 8 will mean that eachmacrocell covers 8x8x8
      cells, and a 128^3 volume (with 127^3 cells...) will this have a
      macrocell grid of 16^3 macro cells (out of which the
      right/back/top one will only have 7x7x7 of its 8x8x8 covered by
      actual cells */
  enum { cellsPerMC = 8 };

  /*! compute kernel that computes macro-cell information for a 3D
      structured data grid */
  struct StructuredData_ComputeMCs {
#if RTC_DEVICE_CODE
    /* kernel CODE */
    inline __rtc_device void run(const rtc::ComputeInterface &rtCore)
    {
      vec3i mcID
        = vec3i(rtCore.getThreadIdx())
        + vec3i(rtCore.getBlockIdx())
        * vec3i(rtCore.getBlockDim());
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
            float f = rtc::tex3D<float>(scalars,
                                   (float)scalarID.x,
                                   (float)scalarID.y,
                                   (float)scalarID.z);
            scalarRange.extend(f);
          }
      int mcIdx = mcID.x + mcGrid.dims.x*(mcID.y+mcGrid.dims.y*(mcID.z));
      mcGrid.scalarRanges[mcIdx] = scalarRange;
    }
#endif      
    /* kernel ARGS */
    MCGrid::DD mcGrid;
    vec3i numScalars;
    rtc::TextureObject scalars;
  };
  
  StructuredData::StructuredData(Context *context,
                                 const DevGroup::SP &devices)
    : ScalarField(context,devices)
  {
    perLogical.resize(devices->numLogical);
    for (auto device : *devices)
      getPLD(device)->computeMCs
        = createCompute_StructuredData_computeMCs(device->rtc);
  }


  void StructuredData::buildMCs(MCGrid &mcGrid) 
  {
    vec3i mcDims = divRoundUp(numCells,vec3i(cellsPerMC));
    mcGrid.resize(mcDims);
    vec3ui blockSize(4);
    vec3ui numBlocks = divRoundUp(vec3ui(mcDims),blockSize);
    mcGrid.gridOrigin = worldBounds.lower;
    mcGrid.gridSpacing = vec3f(cellsPerMC) * this->gridSpacing;
    for (auto device : *devices) {
      PLD *pld = getPLD(device);
      StructuredData_ComputeMCs args = {
        mcGrid.getDD(device),
        numScalars,
        textureNN->getDD(device)
      };
      pld->computeMCs->launch(numBlocks,blockSize,
                              &args);
    }
    for (auto device : *devices)
      device->sync();
  }
  
  StructuredDataSampler::DD StructuredDataSampler::getDD(Device *device)
  {
    DD dd;
    dd.texObj = sf->texture->getDD(device);
    dd.cellGridOrigin = sf->gridOrigin;
    dd.cellGridSpacing = sf->gridSpacing;
    dd.numCells = sf->numCells;
    return dd;
  }
  
  VolumeAccel::SP StructuredData::createAccel(Volume *volume) 
  {
    auto sampler = std::make_shared<StructuredDataSampler>(this);
    return std::make_shared<MCVolumeAccel<StructuredDataSampler>>
      (volume,
       createGeomType_StructuredMC,
       sampler);
  }
  
  // ==================================================================
  bool StructuredData::set3f(const std::string &member,
                             const vec3f &value) 
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
  bool StructuredData::set3i(const std::string &member,
                             const vec3i &value) 
  {
    if (member == "dims") {
      numScalars = value;
      numCells   = value - vec3i(1);
      return true;
    }
    return false;
  }

  // ==================================================================
  bool StructuredData::setObject(const std::string &member,
                                 const Object::SP &value) 
  {
    if (member == "textureData") {
      scalars = value->as<TextureData>();
      BNTextureAddressMode addressModes[3] = {
        BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP,BN_TEXTURE_CLAMP
      };
      texture = std::make_shared<Texture>((Context*)context,scalars,
                                          BN_TEXTURE_LINEAR,
                                          addressModes,
                                          BN_COLOR_SPACE_LINEAR);
      textureNN = std::make_shared<Texture>((Context*)context,scalars,
                                            BN_TEXTURE_NEAREST,
                                            addressModes,
                                            BN_COLOR_SPACE_LINEAR);
      return true;
    }
    return false;
  }

  // ==================================================================
  void StructuredData::commit() 
  {
    worldBounds.lower = gridOrigin;
    worldBounds.upper = gridOrigin + gridSpacing * vec3f(numCells);
  }
  
  RTC_EXPORT_COMPUTE3D(StructuredData_computeMCs,StructuredData_ComputeMCs);
}

