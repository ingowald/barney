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

/* the "multipass" functionality allows certain objects to be traced
   against in a separate pass; i.e., _not_ in an intersection program
   of the main scene, but with its 'traceRays()' function that can
   then do whatever it wants, including laucnhing a separate raygen
   program, etc. THis is obviously highly 'disruptive' to the main
   design of barney - and frankly, to how a ray tracer _should_ work
   (in particular, it means that such objects are intersected "out of
   traversal order", potentially leading to wasteful intersections,
   traversals, or even launches) - but on the uptime, it allows such
   objects to use the entire gpu (including ray tracing cores etc) in
   whatever way it deems fit (e.g., it'll allow a umesh to use the rt
   cores for cell location, which it cannot easily do from within an
   intersection pgoram of the 'main' world). As such: this
   functionality may be useful - but "handle with care" */
#pragma once

#include "barney/geometry/Geometry.h"
#include "barney/volume/Volume.h"

namespace barney {

  struct DataGroup;
  struct Group;
  
  struct MultiPass {
    struct Object {
      typedef std::shared_ptr<Object> SP;
      virtual void traceRays(RayQueue &rays, const affine3f &xfm) = 0;
    };
    
    struct Instances : public std::vector<std::pair<MultiPass::Object::SP,affine3f>>
    { 
      void instantiate(Group *, const affine3f &xfm);
    };
  };
  
}
