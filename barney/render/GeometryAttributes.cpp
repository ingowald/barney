// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/render/GeometryAttributes.h"

namespace BARNEY_NS {
  namespace render {
      
    GeometryAttributes::DD GeometryAttributes::getDD(Device *device)
    {
      GeometryAttributes::DD attributes;
      auto set = [&](GeometryAttribute::DD &out,
                     const GeometryAttribute &in,
                     const std::string &dbgName)
      {
        if (in.faceVarying) {
          out.scope = GeometryAttribute::FACE_VARYING;
          out.fromArray.type = in.faceVarying->type;
          out.fromArray.ptr
            = in.faceVarying->getDD(device);
          out.fromArray.size = (int)in.faceVarying->count;
        } else if (in.perVertex) {
          out.scope = GeometryAttribute::PER_VERTEX;
          out.fromArray.type = in.perVertex->type;
          out.fromArray.ptr
            = in.perVertex->getDD(device);
          out.fromArray.size = (int)in.perVertex->count;
        } else if (in.perPrim) {
          out.scope = GeometryAttribute::PER_PRIM;
          out.fromArray.type = in.perPrim->type;
          out.fromArray.ptr
            = in.perPrim->getDD(device);
          out.fromArray.size = (int)in.perPrim->count;
        } else if (isnan(in.constant[0])) {
          out.scope = GeometryAttribute::INVALID;
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
      set(attributes.normalAttribute,
          this->normalAttribute,"normal");
      return attributes;
    }
    
  }
}
