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

#include "barney/LocalFB.h"
#include "barney/Model.h"

namespace barney {

  Model::Model(Context *context)
    : context(context)
  {
    for (int localID=0;localID<context->perDG.size();localID++) {
      dataGroups.push_back(DataGroup::create(this,localID));
    }
  }

  void Model::render(const Camera *camera,
                     FrameBuffer *fb)
  {
    assert(context);
    assert(fb);
    assert(camera);
    context->ensureRayQueuesLargeEnoughFor(fb);
    context->render(this,camera,fb);
  }

}
