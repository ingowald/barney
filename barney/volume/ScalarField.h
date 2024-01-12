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

#include <array>

#include "barney/volume/Volume.h"

namespace barney {

  struct Volume;
  struct VolumeAccel;
  struct MCGrid;

  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public Object
  {
    typedef std::shared_ptr<ScalarField> SP;

    struct DD {
      box4f             worldBounds;
    };
    
    ScalarField(DevGroup *devGroup)
      : devGroup(devGroup)
    {}

    OWLContext getOWL() const;
    // virtual std::vector<OWLVarDecl> getVarDecls(uint32_t baseOfs) = 0;
    virtual void setVariables(OWLGeom geom);
    
    virtual std::shared_ptr<VolumeAccel> createAccel(Volume *volume) = 0;
    virtual void buildMCs(MCGrid &macroCells)
    { throw std::runtime_error("this calar field type does not know how to build macro-cells"); }

    /*! returns (part of) a string that should allow an OWL geometry
        type to properly create all the names of all the optix device
        functions that operate on this type. Eg, if all device
        functions for a "StucturedVolume" are named
        "Structured_<SomeAccel>_{Bounds,Isec,CV}()", then the
        StructuredField should reutrn "Structured", and somebody else
        can/has to then make sure to add the respective
        "_<SomeAccel>_" part. */
    virtual std::string getTypeString() const { BARNEY_NYI(); }
        
    DevGroup *const devGroup;
    box4f     worldBounds;
  };
  
}

