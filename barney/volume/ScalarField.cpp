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

#include "barney/volume/Volume.h"
#include "barney/volume/ScalarField.h"
#include "barney/ModelSlot.h"
#include "barney/Context.h"
#include "barney/volume/StructuredData.h"

namespace barney {

  void ScalarField::buildMCs(MCGrid &macroCells)
  { throw std::runtime_error("this calar field type does not know how to build macro-cells"); }
  
  void ScalarField::setVariables(OWLGeom geom)
  {
    box3f bb = worldBounds;
    owlGeomSet3fv(geom,"worldBounds.lower",&bb.lower.x);
    owlGeomSet3fv(geom,"worldBounds.upper",&bb.upper.x);
  }

  ScalarField::ScalarField(Context *context, int slot,
                           const box3f &domain)
    : SlottedObject(context,slot),
      domain(domain)
  {}

  void ScalarField::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back
      ({"worldBounds.lower",OWL_FLOAT3,base+OWL_OFFSETOF(DD,worldBounds.lower)});
    vars.push_back
      ({"worldBounds.upper",OWL_FLOAT3,base+OWL_OFFSETOF(DD,worldBounds.upper)});
  }

  ScalarField::SP ScalarField::create(Context *context,
                                      int slot,
                                      const std::string &type)
  {
    if (type == "structured")
      return std::make_shared<StructuredData>(context,slot);
    
    context->warn_unsupported_object("ScalarField",type);
    return {};
  }

}
