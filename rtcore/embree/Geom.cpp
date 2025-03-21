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

#include "rtcore/embree/Geom.h"

namespace rtc {
  namespace embree {
    
    Geom::Geom(GeomType *type)
      : type(type),
        programData(type->sizeOfProgramData)
    {}
    
    void Geom::setDD(const void *dd)
    {
      memcpy(programData.data(),dd,programData.size());

      uint8_t *ptr = (uint8_t*)programData.data();
      ptr += programData.size();
    }

  }
}

