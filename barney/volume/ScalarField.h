// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <array>

#include "barney/volume/MCGrid.h"
#include "barney/volume/Volume.h"

namespace BARNEY_NS {

  struct Volume;
  struct VolumeAccel;
  struct IsoSurface;
  struct IsoSurfaceAccel;
  struct ModelSlot;

  /*! abstracts any sort of scalar field (unstructured, amr,
    structured, rbfs....) _before_ any transfer function(s) get
    applied to it */
  struct ScalarField : public barney_api::ScalarField
  {
    typedef std::shared_ptr<ScalarField> SP;

    /*! Device-side data common to all ScalarFields that live on the device */
    struct DD {
      /*! world bounds, CLIPPED TO DOMAIN (if non-empty domain is present!) */
      box3f                worldBounds;
    };
    DD getDD(Device *device) const { return { worldBounds }; }
    
    ScalarField(Context *context,
                const DevGroup::SP &devices,
                const box3f &domain=box3f());

    static ScalarField::SP create(Context *context,
                                  const DevGroup::SP &devices,
                                  const std::string &type);

    /*! creates an acceleration structure for a 'volume' object using
        this scalar field type */
    virtual std::shared_ptr<VolumeAccel>
    createAccel(Volume *volume) = 0;

    /*! creates an acceleration structure for a 'isoSurface' geometry
        using this scalar field type */
    virtual std::shared_ptr<IsoSurfaceAccel>
    createIsoAccel(IsoSurface *isoSurface)
    { return {}; }

    MCGrid::SP getMCs()
    { if (!mcGrid) mcGrid = buildMCs(); return mcGrid; }

    /*! create, fill, and return a macrocell grid for this field */
    virtual MCGrid::SP buildMCs();

    MCGrid::SP  mcGrid;
    box3f       worldBounds;
    
    /*! a clipping box used to restrict whatever primitives the volume
        may be made up to down to a specific 3d box. e.g., if there's
        ghost cells, or if this is a spatial partitioning of a umesh,
        etc */
    const box3f domain;
    DevGroup::SP const devices;
  };
  
  /*! abstraction for a class that can sample a given scalar
    field. it's up to that class to create the right sampler for its
    data, and to do that only for the kind of traversers/accels that
    actually need to be able to sample.

    For the device side, all the actual sampling functionality will be
    in the DD's of the derived classes; the parent class doesnt' even
    have a DD, because all the device-side sampling code will (have to
    be) resolved through templates, in which case the classes using
    the sampler will know the actual type of that sampler (and it's
    DD)
  */
  struct ScalarFieldSampler {
    virtual void build() = 0;
    struct DD {
      /* derived classes ned to implement:
         
         inline __both__ float sample(vec3f P, bool dbg)
         
      */
    };
  };
  
}

