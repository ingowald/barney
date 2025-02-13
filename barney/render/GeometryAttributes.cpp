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

#include "barney/render/GeometryAttributes.h"

namespace barney {
  namespace render {
      
    GeometryAttributes::DD GeometryAttributes::getDD(Device *device)
    {
      GeometryAttributes::DD attributes;
      auto set = [&](GeometryAttribute::DD &out,
                     const GeometryAttribute &in,
                     const std::string &dbgName)
      {
        if (in.perVertex) {
          out.scope = GeometryAttribute::PER_VERTEX;
          out.fromArray.type = in.perVertex->type;
          out.fromArray.ptr
            // = owlBufferGetPointer(in.perVertex->owl,devID);
            = in.perVertex->getDD(device);
          out.fromArray.size = (int)in.perVertex->count;
        } else if (in.perPrim) {
          out.scope = GeometryAttribute::PER_PRIM;
          out.fromArray.type = in.perPrim->type;
          out.fromArray.ptr
            // = owlBufferGetPointer(in.perPrim->owl,devID);
            = in.perPrim->getDD(device);
          out.fromArray.size = (int)in.perPrim->count;
        } else {
          out.scope = GeometryAttribute::CONSTANT;
          (vec4f&)out.value = in.constant;
        }
      };

      for (int i=0;i<attributes.count;i++) {
        auto &out = attributes.attribute[i];
        const auto &in = this->attribute[i];
        set(out,in,"attr"+std::to_string(i));
      }
      set(attributes.colorAttribute,
          this->colorAttribute,"color");
      return attributes;
    }
    
  }
}
