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

#include "barney/volume/TransferFunction.h"

namespace barney {

  TransferFunction::TransferFunction(DevGroup *devGroup)
    : devGroup(devGroup)
  {
    domain = { 0.f,1.f };
    values = { vec4f(1.f), vec4f(1.f) };
    baseDensity  = 1.f;
    valuesBuffer = owlDeviceBufferCreate(devGroup->owl,
                                         OWL_FLOAT4,
                                         values.size(),
                                         values.data());
  }

  void TransferFunction::DD::addVars(std::vector<OWLVarDecl> &vars, int base)
  {
    vars.push_back({"xf.values",OWL_BUFPTR,base+OWL_OFFSETOF(DD,values)});
    vars.push_back({"xf.numValues",OWL_INT,base+OWL_OFFSETOF(DD,numValues)});
    vars.push_back({"xf.baseDensity",OWL_FLOAT,base+OWL_OFFSETOF(DD,baseDensity)});
    vars.push_back({"xf.domain",OWL_FLOAT2,base+OWL_OFFSETOF(DD,domain)});
  }
  
  void TransferFunction::set(const range1f &domain,
                             const std::vector<vec4f> &values,
                             float baseDensity)
  {
    this->domain = domain;
    this->baseDensity = baseDensity;
    this->values = values;
    owlBufferResize(this->valuesBuffer,this->values.size());
    owlBufferUpload(this->valuesBuffer,this->values.data());
  }

    /*! get cuda-usable device-data for given device ID (relative to
        devices in the devgroup that this gris is in */
  TransferFunction::DD TransferFunction::getDD(int devID) const
  {
    TransferFunction::DD dd;

    dd.values = (float4*)owlBufferGetPointer(valuesBuffer,devID);
    dd.domain = domain;
    dd.baseDensity = baseDensity;
    dd.numValues = (int)values.size();

    return dd;
  }
    
  std::vector<OWLVarDecl> TransferFunction::getVarDecls(uint32_t myOffset)
  {
    return
      {
       { "xf.values",      OWL_BUFPTR, myOffset+OWL_OFFSETOF(DD,values) },
       { "xf.domain",      OWL_FLOAT2, myOffset+OWL_OFFSETOF(DD,domain) },
       { "xf.baseDensity", OWL_FLOAT,  myOffset+OWL_OFFSETOF(DD,baseDensity) },
       { "xf.numValues",   OWL_INT,    myOffset+OWL_OFFSETOF(DD,numValues) },
      };
  }
  
  void TransferFunction::setVariables(OWLGeom geom) const
  {
    // if (!(domain.lower < domain.upper)) 
    //   throw std::runtime_error("in-valid domain for transfer function");
    
    owlGeomSet2f(geom,"xf.domain",domain.lower,domain.upper);
    owlGeomSet1f(geom,"xf.baseDensity",baseDensity);
    owlGeomSet1i(geom,"xf.numValues",(int)values.size());
    // intentionally set to null for first-time build
    owlGeomSetBuffer(geom,"xf.values",valuesBuffer);
  }
  
  
}
