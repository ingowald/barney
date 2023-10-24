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

#pragma once

#include "barney/Object.h"

namespace barney {

  struct DataGroup;
  struct Volume;

  typedef std::array<int,4> TetIndices;
  typedef std::array<int,5> PyrIndices;
  typedef std::array<int,6> WedIndices;
  typedef std::array<int,8> HexIndices;

  struct ScalarField : public Object
  {
    typedef std::shared_ptr<ScalarField> SP;
    
    ScalarField(DataGroup *owner)
      : owner(owner)
    {}

    OWLContext getOWL() const;
    
    virtual void build(Volume *) {}
    
    DataGroup *owner;
  };

  // struct TransferFunction : public Object
  // {
  //   typedef std::shared_ptr<TransferFunction> SP;
    
  //   TransferFunction(DataGroup *owner,
  //                    const range1f &domain,
  //                    const std::vector<float4> &values,
  //                    float baseDensity);

  //   /*! pretty-printer for printf-debugging */
  //   std::string toString() const override
  //   { return "TransferFunction{}"; }

    // range1f             domain;
    // std::vector<float4> values;
    // OWLBuffer           valuesBuffer;
    // float               baseDensity;
  //   DataGroup *owner;
  // };
    
  struct Volume : public Object
  {
    typedef std::shared_ptr<Volume> SP;
    
    Volume(DataGroup *owner,
           ScalarField::SP sf);

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Volume{}"; }

    virtual void build()
    { sf->build(this); }
    
    DataGroup  *owner;
    
    void setXF(const range1f &domain,
               const std::vector<vec4f> &values,
               float baseDensity);
               
    ScalarField::SP      sf;

    struct {
      range1f             domain;
      std::vector<vec4f>  values;
      OWLBuffer           valuesBuffer;
      float               baseDensity;
    } xf;

    std::vector<OWLGroup> generatedGroups;
    // std::vector<OWLGeom>  userGeoms;
    // std::vector<OWLGeom>  triangleGeoms;
  };

}
