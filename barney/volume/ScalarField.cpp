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
#include "barney/umesh/common/UMeshField.h"

namespace BARNEY_NS {

  void ScalarField::buildMCs(MCGrid &macroCells)
  {
    throw std::runtime_error
      ("this calar field type does not know how to build macro-cells");
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
    
    context->warn_unsupported_object("ScalarField",type);
    return {};
  }

}
