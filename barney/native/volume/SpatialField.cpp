// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/Volume.h"
#include "native/volume/SpatialField.h"
// #include "native/volume/SpatialFieldRegistry.h"
#include "native/ModelSlot.h"
#include "native/Context.h"
#include "native/volume/StructuredData.h"
#include "native/umesh/common/UMeshField.h"
#include "native/amr/BlockStructuredField.h"
#include "native/volume/NanoVDB.h"

namespace BARNEY_NS {
  namespace native {

    MCGrid::SP SpatialField::buildMCs()
    {
      throw std::runtime_error
        ("this calar field type does not know how to build macro-cells");
      return {};
    }
  
    SpatialField::SpatialField(Device *device,
                               const box3f &domain)
      : Object(device),
        domain(domain)
    {}

    /*! pretty-printer for printf-debugging */
    std::string SpatialField::toString() const
    { return "barney::native::SpatialField"; }
    
    SpatialField::SP SpatialField::create(Device *device,
                                          const std::string &type)
    {
      if (type == "NanoVDB") {
#if BARNEY_HAVE_NANOVDB
        return std::make_shared<NanoVDBData>(device);
#else
        throw std::runtime_error("NanoVDB geometry type not enabled in this build");
#endif
      }
      if (type == "structured")
        return std::make_shared<StructuredData>(device); 

      if (type == "unstructured")
        return std::make_shared<UMeshField>(device); 

      if (type == "BlockStructuredAMR")
        return std::make_shared<BlockStructuredField>(device); 
    
      return {};
    }

  }
}
