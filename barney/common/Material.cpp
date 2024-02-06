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

#include "barney/common/Material.h"
#include "barney/DataGroup.h"

namespace barney {
  
  void Material::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back({"material.baseColor", OWL_FLOAT3, base+OWL_OFFSETOF(DD,baseColor)});
    vars.push_back({"material.alphaTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,alphaTexture)});
    vars.push_back({"material.colorTexture", OWL_TEXTURE, base+OWL_OFFSETOF(DD,colorTexture)});
  }

  void Material::set(OWLGeom geom) const
  {
    owlGeomSet3f(geom,"material.baseColor",
                 baseColor.x,
                 baseColor.y,
                 baseColor.z);
    owlGeomSet1f(geom,"transmission",transmission);
    owlGeomSet1f(geom,"ior",ior);
    owlGeomSetTexture(geom,"alphaTexture",
                      alphaTexture
                      ? alphaTexture->owlTex
                      : (OWLTexture)0
                      );
    owlGeomSetTexture(geom,"colorTexture",
                      colorTexture
                      ? colorTexture->owlTex
                      : (OWLTexture)0
                      );
  }

}
