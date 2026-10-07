// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/native/Context.h"
#include "barney/native/geometry/Geometry.h"
#include "cuBQL/bvh.h"

namespace BARNEY_NS {
  namespace splats {
    namespace native {
      using namespace BARNEY_NS::native;
      
      struct Splats : public Geometry
      {
        typedef std::shared_ptr<Splats> SP;

        struct PLD {
          cuBQL::bvh3f bvh;
        };
        PLD *getPLD(Device *device);
        std::vector<PLD> perLogical;
        
        Splats(native::Context *context, int slot);
        ~Splats() override;
        
        /*! pretty-printer for printf-debugging */
        std::string toString() const override;
        
        /*! ask this geometry to build whatever owl geoms it needs to build */
        void build() override;

        PODData::SP indexData;
        PODData::SP vertexPositionData;
        PODData::SP vertexRadiusData;
        float globalRadius{0.f};
        
      private:
        void free();
      };
      
    }
  }
}
