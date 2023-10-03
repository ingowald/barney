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

#include "barney/Context.h"

namespace barney {

  struct Model : public Object {
    typedef std::shared_ptr<Model> SP;

    static SP create(Context *ctx) { return std::make_shared<Model>(ctx); }
    
    Model(Context *context)
      : context(context)
    {}
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Model{}"; }
    
    void render(const BNCamera *camera,
                FrameBuffer *fb,
                uint32_t *appfB);
    
    Context *const context;
  };

}
