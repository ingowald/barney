// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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
#include "mori/DeviceGroup.h"

namespace barney {
  
  Context::Context(const std::vector<int> &dataGroupIDs,
                   const std::vector<int> &gpuIDs)
    : dataGroupIDs(dataGroupIDs),
      gpuIDs(gpuIDs)
  {
    if (gpuIDs.size() < dataGroupIDs.size())
      throw std::runtime_error("not enough GPUs ("
                               +std::to_string(gpuIDs.size())
                               +") for requested num data groups ("
                               +std::to_string(gpuIDs.size())
                               +")");
    if (gpuIDs.size() % dataGroupIDs.size())
      throw std::runtime_error("requested num GPUs is not a multiple of "
                               "requested num data groups");
    int numMoris = dataGroupIDs.size();
    int gpusPerMori = gpuIDs.size() / numMoris;
    moris.resize(numMoris);
    for (int moriID=0;moriID<numMoris;moriID++) {
      std::vector<int> gpusThisMori(gpusPerMori);
      for (int j=0;j<gpusPerMori;j++)
        gpusThisMori[j] = gpuIDs[moriID*gpusPerMori+j];
      moris[moriID] = mori::DeviceGroup::create(gpusThisMori);
    }
  }
      
}

