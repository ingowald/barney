// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "barney/geometry/Geometry.h"
#include "barney/DataGroup.h"

namespace barney {
  
  OWLContext Geometry::getOWL() const
  { return owner->getOWL(); }

  void Geometry::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back({"material.baseColor", OWL_FLOAT3, base+OWL_OFFSETOF(DD,material.baseColor)});
    vars.push_back({"material.alphaTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,material.alphaTexture)});
    vars.push_back({"material.colorTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,material.colorTexture)});
  }
  
  void Geometry::setMaterial(OWLGeom geom)
  {
    owlGeomSet3f(geom,"material.baseColor",
                 material.baseColor.x,
                 material.baseColor.y,
                 material.baseColor.z);
    owlGeomSetTexture(geom,"material.alphaTexture",
                      material.alphaTexture
                      ? material.alphaTexture->owlTex
                      : (OWLTexture)0
                      );
    owlGeomSetTexture(geom,"material.colorTexture",
                      material.colorTexture
                      ? material.colorTexture->owlTex
                      : (OWLTexture)0
                      );
  }
  
}

