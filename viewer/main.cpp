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

#include "barney.h"
#include "mpiInf/Renderer.h"
#if CUTEE
# include "qtOWL/OWLViewer.h"
# include "qtOWL/XFEditor.h"
#else
# include "samples/common/owlViewer/InspectMode.h"
# include "samples/common/owlViewer/OWLViewer.h"
#endif

namespace bo {
  
  struct SphereSet {
    struct Sphere {
      float3   position;
      half     radius;
      uint16_t type;
    };
    std::vector<Sphere> spheres;
  }
      
  struct BarnOWL {

    struct RankData {
      mini::Scene::SP               triangles;
      SphereSet                     spheres;
      std::vector<umesh::UMesh::SP> unsts;
    };

    BarnOWN(const std::vector<RankData> &data,
            BarnComm &comm);
    const std::vector<RankData> rankData;
  };
  
}

using namespace bo;

int main(int ac, char **av)
{
    mpiInf::Comms comms(ac,(char **&)av);
}
