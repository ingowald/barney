// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/volume/ScalarField.h"
#include <functional>
#include <map>
#include <string>
#include <mutex>

namespace BARNEY_NS {

  // Factory function type for creating scalar fields
  using ScalarFieldFactory = std::function<ScalarField::SP(Context*, const DevGroup::SP&)>;

  // Global registry for scalar field types
  // Allows external plugins to register custom field types at static initialization time
  class ScalarFieldRegistry {
  public:
    static ScalarFieldRegistry& instance() {
      static ScalarFieldRegistry registry;
      return registry;
    }
    
    // Register a new scalar field type
    void registerType(const std::string& type, ScalarFieldFactory factory) {
      std::lock_guard<std::mutex> lock(mutex_);
      factories_[type] = factory;
    }
    
    // Create a scalar field by type
    ScalarField::SP create(Context* context, const DevGroup::SP& devices, const std::string& type) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = factories_.find(type);
      if (it != factories_.end()) {
        return it->second(context, devices);
      }
      return nullptr;
    }
    
    // Check if a type is registered
    bool hasType(const std::string& type) const {
      std::lock_guard<std::mutex> lock(mutex_);
      return factories_.find(type) != factories_.end();
    }
    
  private:
    ScalarFieldRegistry() = default;
    ScalarFieldRegistry(const ScalarFieldRegistry&) = delete;
    ScalarFieldRegistry& operator=(const ScalarFieldRegistry&) = delete;
    
    std::map<std::string, ScalarFieldFactory> factories_;
    mutable std::mutex mutex_;
  };

  // Helper macro for registration (use in .cpp files)
  // Creates a static object that registers the field type before main() runs
  #define BARNEY_REGISTER_SCALAR_FIELD(TYPE_NAME, CLASS_NAME) \
    namespace { \
      struct CLASS_NAME##Registrar { \
        CLASS_NAME##Registrar() { \
          ScalarFieldRegistry::instance().registerType(TYPE_NAME, \
            [](Context* ctx, const DevGroup::SP& devs) -> ScalarField::SP { \
              return std::make_shared<CLASS_NAME>(ctx, devs); \
            }); \
        } \
      }; \
      static CLASS_NAME##Registrar g_##CLASS_NAME##Registrar; \
    }

} // namespace BARNEY_NS
