// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/common/barney-common.h"
#include "barney/Object.h"
#include "barney/volume/TransferFunction.h"
#include "barney/volume/ScalarField.h"
#include <array>

namespace BARNEY_NS {

  using Random = Random2;
  struct Volume;
  struct ScalarField;
  struct MCGrid;
  
  typedef std::array<int,4> TetIndices;
  typedef std::array<int,5> PyrIndices;
  typedef std::array<int,6> WedIndices;
  typedef std::array<int,8> HexIndices;

  struct VolumeAccel {
    
    typedef std::shared_ptr<VolumeAccel> SP;

    VolumeAccel(Volume *volume);

    virtual void build(bool full_rebuild) = 0;

    const TransferFunction *getXF() const;
    
    Volume      *const volume = 0;
    const DevGroup::SP devices;
  };





  struct VolumeAccel;
  
  /*! a *volume* is a scalar field with a transfer function applied to
      it; it's main job is to create something that can intersect a
      ray with that scalars-plus-transferfct thingy, for which it will
      use some kind of volume accelerator that implements the
      scalar-field type specific stuff (eg, traverse a bvh over
      elements, or look up a 3d texture, etc) */
  struct Volume : public barney_api::Volume
  {
    template<typename SFSampler>
    struct DD {
      inline __rtc_device
      vec4f sampleAndMap(vec3f point, bool dbg=false) const
      {
        float f = sfSampler.sample(point,dbg);
        if (isnan(f)) return vec4f(0.f);
        vec4f mapped = xf.map(f,dbg);
        return mapped;
      }
      
      ScalarField::DD        sfCommon;
      typename SFSampler::DD sfSampler;
      TransferFunction::DD   xf;
      int                    userID;
    };
    
    template<typename SFSampler>
    DD<SFSampler> getDD(Device *device, std::shared_ptr<SFSampler> sampler)
    {
      DD<SFSampler> dd;
      dd.sfCommon = sf->getDD(device);
      dd.sfSampler = sampler->getDD(device);
      dd.xf = xf.getDD(device);
      dd.userID = userID;
      return dd;
    }

    typedef std::shared_ptr<Volume> SP;
    
    Volume(ScalarField::SP sf);

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Volume{}"; }

    static SP create(ScalarField::SP sf)
    {
      return std::make_shared<Volume>(sf);
    }
    
    /*! (re-)build the accel structure for this volume, probably after
        changes to transfer functoin (or later, scalar field) */
    virtual void build(bool full_rebuild);
    
    void setXF(const range1f &domain,
               const bn_float4 *values,
               int numValues,
               float baseDensity) override;
    bool set1i(const std::string &member,
               const int   &value) override;
               
    ScalarField::SP  sf;
    VolumeAccel::SP  accel;
    TransferFunction xf;
    DevGroup::SP const devices;
    int userID = 0;
    
    struct PLD {
      std::vector<rtc::Group *> generatedGroups;
      std::vector<rtc::Geom *>  generatedGeoms;
    };
    PLD *getPLD(Device *device);
    std::vector<PLD> perLogical;
  };


  inline __rtc_device float fastLog(float f)
  {
    f = (f-1.f)/(f+1.f);
    float f2 = f*f;
    float s = f;
    f *= f2;
    s += (1.f/3.f)*f;
    f *= f2;
    s += (1.f/5.f)*f;
    return s+s;
  }
  
  /*! helper class that performs woodcock sampling over a given
      parameter range, for a given sample'able volume type */
  struct Woodcock {
    template<typename VolumeDD>
    static inline __rtc_device
    bool sampleRange(vec4f &sample,
                     const VolumeDD &sfSampler,
                     vec3f org, vec3f dir,
                     range1f &tRange,
                     float majorant,
                     Random &rand,
                     bool dbg=false) 
    {
      float t = tRange.lower;
      while (true) {
        float r = rand();
        float dt = - fastLog(1.f-r)/majorant;
        // float dt = - logf(1.f-r)/majorant;
        t += dt;
        if (t >= tRange.upper)
          return false;

        vec3f P = org+t*dir;
        sample = sfSampler.sampleAndMap(P,dbg);
        // if (dbg) printf("sample at t %f, P= %f %f %f -> %f %f %f : %f\n",
        //                 t,
        //                 P.x,P.y,P.z,
        //                 sample.x,
        //                 sample.y,
        //                 sample.z,
        //                 sample.w);
        if (sample.w >= rand()*majorant) {
          tRange.upper = t;
          return true;
        }
      }
    }
  };

  inline VolumeAccel::VolumeAccel(Volume *volume)
    : volume(volume),
      devices(volume->devices)
  {
    assert(volume);
    assert(volume->sf);
  }
}
