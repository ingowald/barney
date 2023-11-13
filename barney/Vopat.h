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

#pragma once

#include "barney/Object.h"

namespace barney {

#if VOPAT
  
  struct Proxy {
    box3f bounds;
    int   rank;
    int   localDG;
  };

  struct Shard : public Proxy {
    range1f valueRange;
  };

  struct ProxyGeom {
    struct DD {
      Proxy *proxies;
    };
    OWLGeom            geom = 0;
    OWLBuffer          buffer = 0;
    std::vector<Shard> shards;
  };
  
  struct ShardsGeom {
    struct DD {
      Shard *shards;
    };
    OWLGeom            geom = 0;
    OWLBuffer          buffer = 0;
    std::vector<Shard> shards;
  };
  
  
#endif
}
