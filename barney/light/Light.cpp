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

#include "barney/light/Light.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"

#include "barney/light/QuadLight.h"
#include "barney/light/DirLight.h"
#include "barney/light/EnvMap.h"

namespace barney {

  Light::SP Light::create(ModelSlot *owner,
                          const std::string &type)
  {
    if (type == "directional")
      return std::make_shared<DirLight>(owner);
    if (type == "quad")
      return std::make_shared<QuadLight>(owner);
    if (type == "envmap")
      return std::make_shared<EnvMapLight>(owner);
    
    owner->context->warn_unsupported_object("Light",type);
    return {};
  }

  // ==================================================================
  
  // ==================================================================
  
  // ==================================================================
  
}
