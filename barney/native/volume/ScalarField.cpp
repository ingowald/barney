// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/volume/Volume.h"
#include "native/volume/ScalarField.h"
#include "native/volume/ScalarFieldRegistry.h"
#include "native/ModelSlot.h"
#include "native/Context.h"
#include "native/volume/StructuredData.h"
#include "native/umesh/common/UMeshField.h"
#include "native/amr/BlockStructuredField.h"
#include "native/volume/NanoVDB.h"

namespace BARNEY_NS {
  namespace native {

    MCGrid::SP ScalarField::buildMCs()
    {
      throw std::runtime_error
        ("this calar field type does not know how to build macro-cells");
      return {};
    }
  
    ScalarField::ScalarField(Context *context,
                             const DevGroup::SP &devices,
                             const box3f &domain)
      : Object(context),
        devices(devices),
        domain(domain)
    {}

    // Register core barney scalar field types
    namespace {
      void registerBuiltinTypes() {
        static bool registered = false;
        if (registered) return;
        registered = true;
      
        auto& registry = ScalarFieldRegistry::instance();
      
        registry.registerType("structured", 
                              [](Context* ctx, const DevGroup::SP& devs) { 
                                return std::make_shared<StructuredData>(ctx, devs); 
                              });
      
        registry.registerType("unstructured", 
                              [](Context* ctx, const DevGroup::SP& devs) { 
                                return std::make_shared<UMeshField>(ctx, devs); 
                              });
      
        registry.registerType("BlockStructuredAMR", 
                              [](Context* ctx, const DevGroup::SP& devs) { 
                                return std::make_shared<BlockStructuredField>(ctx, devs); 
                              });
      
        registry.registerType("NanoVDB", 
                              [](Context* ctx, const DevGroup::SP& devs) -> ScalarField::SP {
#if BARNEY_HAVE_NANOVDB
                                return std::make_shared<NanoVDBData>(ctx, devs);
#else
                                throw std::runtime_error("NanoVDB geometry type not enabled in this build");
#endif
                              });
      }
    }

    ScalarField::SP ScalarField::create(Context *context,
                                        const DevGroup::SP &devices,
                                        const std::string &type)
    {
      // Ensure built-in types are registered
      registerBuiltinTypes();
    
      // Try to create from registry (includes both built-in and plugin types)
      auto field = ScalarFieldRegistry::instance().create(context, devices, type);
      if (field) {
        return field;
      }
    
      context->warn_unsupported_object("ScalarField",type);
      return {};
    }

  }
}
