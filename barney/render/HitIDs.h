// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

namespace BARNEY_NS {

  /*! HitIDs are used for a special "ID pass" that computes the
      closest intersection along a (primary) ray, WITHOUT any opacity
      or transparency taken into effect. As such it requires its own
      depth value */
  struct HitIDs {
    float depth = INFINITY;
    int primID = -1;
    int instID = -1;
    int objID  = -1;
  };
  
} // ::BARNEY_NS
