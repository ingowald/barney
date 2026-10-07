// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// barney plugin infrastructure
#include "barney/native/Context.h"
#include "barney/anari/Device.h"
// the actual type(s) we want to export:
#include "anari/Splats.h"

namespace BARNEY_NS {
  namespace splats {
    namespace anari {
    // using BARNEY_NS::anari::PluginInfrastructure;    

      void initPlugin()
      {
        std::cout << OWL_TERMINAL_GREEN;
        std::cout << "=======================================================\n";
        std::cout << "initializing plugin 'splats': plugin registers its types\n";
        std::cout << "=======================================================\n";
        std::cout << OWL_TERMINAL_DEFAULT;
      
        PluginInfrastructure *pi = PluginInfrastructure::get();
        pi->exportGeometry(/* note this will 'overwrite' barney's native 'sphere' type */
                           "sphere",
                           splats::anari::createSplatsGeometry
                          );
      }
    
      const int pluginID
      = PluginInfrastructure::get()->registerPlugin("splats",initPlugin);
      
    }
  }
}
