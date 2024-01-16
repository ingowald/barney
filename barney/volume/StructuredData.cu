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

  enum { cellsPerMC = 8 };

  __global__
  void computeMCs(MCGrid::DD mcGrid,vec3i dims,cudaTextureObject_t texNN)
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
          if (scalarID.x >= dims.x) continue;
          if (scalarID.y >= dims.y) continue;
          if (scalarID.z >= dims.z) continue;
          scalarRange.extend(tex3D<float>(texNN,scalarID.x,scalarID.y,scalarID.z));
        }
    int mcIdx = mcID.x + mcGrid.dims.x*(mcID.y+mcGrid.dims.y*(mcID.z));
    mcGrid.scalarRanges[mcIdx] = scalarRange;
  }
  
  StructuredData::StructuredData(DevGroup *devGroup,
                                 const vec3i &dims,
                                 BNScalarType scalarType,
                                 const void *scalars,
                                 const vec3f &gridOrigin,
                                 const vec3f &gridSpacing)
    : ScalarField(devGroup),
      dims(dims),
      scalarType(scalarType),
      rawScalarData(scalars),
      gridOrigin(gridOrigin),
      gridSpacing(gridSpacing)
  {}

  void StructuredData::buildMCs(MCGrid &mcGrid) 
  {
    vec3i mcDims = divRoundUp(dims,vec3i(cellsPerMC));
    mcGrid.resize(mcDims);
    vec3i blockSize(4);
    vec3i numBlocks = divRoundUp(mcDims,blockSize);
    for (int lDevID=0;lDevID<devGroup->size();lDevID++) {
      computeMCs<<<(const dim3&)numBlocks,(const dim3&)blockSize>>>
        (mcGrid.getDD(lDevID),dims,tex3Ds[lDevID].texObjNN);
    }
  }
  
  ScalarField *DataGroup::createStructuredData(const vec3i &dims,
                                               BNScalarType scalarType,
                                               const void *data,
                                               const vec3f &gridOrigin,
                                               const vec3f &gridSpacing)
  {
    StructuredData::SP sf
      = std::make_shared<StructuredData>(devGroup.get(),
                                         dims,scalarType,data,
                                         gridOrigin,gridSpacing);
    return getContext()->initReference(sf);
  }

}

