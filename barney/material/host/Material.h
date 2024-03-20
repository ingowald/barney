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

#pragma once

#include "barney/Object.h"
#include "barney/material/device/Material.h"
#include "barney/common/mat4.h"

namespace barney {
  /*! barney 'virtual' material implementation that takes anari-like
      material paramters, and then builder barney::render:: style
      device materials to be put into the device geometries */
  struct Material : public SlottedObject {
    typedef std::shared_ptr<Material> SP;

    /*! pretty-printer for printf-debugging */
    std::string toString() const override { return "<Material>"; }

    /*! device-data, as a union of _all_ possible device-side
        materials; we have to use a union here because no matter what
        virtual barney::Material gets created on the host, we have to
        have a single struct we put into the OWLGeom/SBT entry, else
        we'd have to have different OWLGeom type for different
        materials .... and possibly even change the actual OWLGeom
        (and even worse, its type) if the assigned material's type
        changes */
    using DD = barney::render::DeviceMaterial;

    Material(ModelSlot *owner) : SlottedObject(owner) {}
    virtual ~Material() = default;

    // ------------------------------------------------------------------
    /*! @{ parameter set/commit interface */
    void commit() override;
    /*! @} */
    // ------------------------------------------------------------------
    static Material::SP create(ModelSlot *dg, const std::string &type);
    
    void setDeviceDataOn(OWLGeom geom) const;
    
    virtual void createDD(DD &dd, int deviceID) const = 0;

    /*! declares the device-data's variables to an owl geom */
    static void addVars(std::vector<OWLVarDecl> &vars, int base);
    
  };

}
