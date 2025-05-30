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

#include "rtcore/optix/Group.h"
#include "rtcore/optix/Device.h"

namespace rtc {
  namespace optix {

    Group::Group(optix::Device *device, OWLGroup owl)
      : device(device),
        owl(owl)
    {}
    
    rtc::AccelHandle Group::getDD() const
    {
      OptixTraversableHandle handle
        = owlGroupGetTraversable(owl,0);
      return (const rtc::AccelHandle &)handle;
    }
    
    void Group::buildAccel()
    {
      owlGroupBuildAccel(owl);
    }
    
    void Group::refitAccel() 
    {
      owlGroupRefitAccel(owl);
    }
      
  }
}
