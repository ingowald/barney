// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/volume/Volume.h"
#include "barney/volume/ScalarField.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"
#include "barney/volume/StructuredData.h"
#include "barney/umesh/common/UMeshField.h"
#include "barney/amr/BlockStructuredField.h"
#include "barney/volume/NanoVDB.h"

namespace BARNEY_NS {

  MCGrid::SP ScalarField::buildMCs()
  {
    throw std::runtime_error
      ("this calar field type does not know how to build macro-cells");
    return {};
  }
  
  ScalarField::ScalarField(Context *context,
                           const DevGroup::SP &devices,
                           const box3f &domain)
    : barney_api::ScalarField(context),
      devices(devices),
      domain(domain)
  {}

  ScalarField::SP ScalarField::create(Context *context,
                                      const DevGroup::SP &devices,
                                      const std::string &type)
  {
    if (type == "structured")
      return std::make_shared<StructuredData>(context,devices);
    if (type == "unstructured")
      return std::make_shared<UMeshField>(context,devices);
    if (type == "BlockStructuredAMR")
      return std::make_shared<BlockStructuredField>(context,devices);
    if (type == "NanoVDB") {
      return std::make_shared<NanoVDBData>(context,devices);
    }
    
    context->warn_unsupported_object("ScalarField",type);
    return {};
  }

}
