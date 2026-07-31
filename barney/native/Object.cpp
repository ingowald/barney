// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/Object.h"
#include "barney/Context.h"
#include "barney/ModelSlot.h"

namespace BARNEY_NS {
  namespace native {

    Object::Object(Context *context)
      : context(context)
    {};
    
    Object::~Object()
    {}
    
    /*! pretty-printer for printf-debugging */
    std::string Object::toString() const
    { return "<Object>"; }

    void Object::warn_unsupported_member(const std::string &type,
                                         const std::string &member)
    {
      static std::set<std::string> alreadyWarned;
      std::string key = toString()+"_"+type+"_"+member;
      if (// context->
          alreadyWarned.find(key) != // context->
          alreadyWarned.end())
        return;
      std::cout << OWL_TERMINAL_RED
                << "#bn: warning - invalid member access. "
                << "Object '" << toString() << "' does not have a member '"<<member<<"'"
                << " of type '"<< type << "'"
                << OWL_TERMINAL_DEFAULT << std::endl;
      // context->
      alreadyWarned.insert(key);
    }

    SlottedObject::SlottedObject(Context *context,
                                 const DevGroup::SP &devices)
      : Object(context),
        devices(devices)
    {
      assert(devices);
      assert(!devices->empty());
    }
 
  }
}

