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

#include <array>

#include "barney/volume/Volume.h"

namespace barney {

  struct Volume;
  struct VolumeAccel;
  struct MCGrid;
  struct ModelSlot;
  
  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public SlottedObject
  {
    typedef std::shared_ptr<ScalarField> SP;

    /*! Device-side data common to all ScalarFields that live on the device */
    struct DD {
      /*! world bounds, CLIPPED TO DOMAIN (if non-empty domain is present!) */
      box3f                worldBounds;

      /*! "template-virtual" function that a sampler calls on an
          _already_ transfer-function mapped RGBA value, allowing the
          scalar field to do some additional color mapping on top of
          whatever came out of the transfer function. the default
          implementation (provided here int his base class coommon to
          all scalar fields) is to just return the xf-color mapped
          RBGA value */
      inline __device__ vec4f mapColor(vec4f xfColorMapped,
                                       vec3f point, float scalar) const
      { return xfColorMapped; }

      static void addVars(std::vector<OWLVarDecl> &vars, int base);
    };
    
    ScalarField(ModelSlot *owner,
                const box3f &domain=box3f());

    static ScalarField::SP create(ModelSlot *dg, const std::string &type);
    
    OWLContext getOWL() const;
    
    virtual void setVariables(OWLGeom geom);
    
    virtual std::shared_ptr<VolumeAccel> createAccel(Volume *volume) = 0;
    
    virtual void buildMCs(MCGrid &macroCells);

    DevGroup *const devGroup;
    box3f     worldBounds;
    
    /*! a clipping box used to restrict whatever primitives the volume
        may be made up to down to a specific 3d box. e.g., if there's
        ghost cells, or if this is a spatial partitioning of a umesh,
        etc */
    const box3f domain;
  };
  
}

