// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "barney/Object.h"
#include "barney/Context.h"
#include "barney/ModelSlot.h"

namespace barney {

  Object::Object(Context *context)
    : context(context)
  {}
  
  void Object::warn_unsupported_member(const std::string &type,
                                       const std::string &member)
  {
    std::string key = toString()+"_"+type+"_"+member;
    if (context->alreadyWarned.find(key) != context->alreadyWarned.end())
      return;
    std::cout << OWL_TERMINAL_RED
              << "#bn: warning - invalid member access. "
              << "Object '" << toString() << "' does not have a member '"<<member<<"'"
              << " of type '"<< type << "'"
              << OWL_TERMINAL_DEFAULT << std::endl;
    context->alreadyWarned.insert(key);
  }
  

  SlottedObject::SlottedObject(Context *context, int slot)
    : Object(context),
      slot(slot)
  {}
 
  DevGroup *SlottedObject::getDevGroup() const
  {
    assert(context);
    DevGroup *dg = context->getDevGroup(slot);
    assert(dg);
    return dg;
  }

  const std::vector<std::shared_ptr<Device>> &SlottedObject::getDevices() const
  {
    return context->getDevices(slot);
    // return getDevGroup()->devices;
  }
  
  OWLContext     SlottedObject::getOWL() const
  {
    if (slot == -1) {
      return context->getOWL(slot);
    }
    return getDevGroup()->owl;
  }
  
}

