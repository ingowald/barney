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

#include "barney/Context.h"
#include "mori/Geometry.h"

namespace barney {

  struct Spheres;
  struct DataGroup;
  
  struct Geom : public Object {
    typedef std::shared_ptr<Geom> SP;

    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Geom{}"; }

    std::vector<mori::Geom::SP> onGPU;
  };

  struct Group : public Object {
    typedef std::shared_ptr<Group> SP;

    static SP create(DataGroup *owner,
                     const std::vector<Geom::SP> &geoms)
    { return std::make_shared<Group>(owner,geoms); }
    
    Group(DataGroup *owner,
          const std::vector<Geom::SP> &geoms);
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Group{}"; }

    void build();
    std::vector<Geom::SP> geoms;

    struct PerGPU {
      mori::TriangleGeomsGroup::SP triangleGeomsGroup;
      mori::UserGeomsGroup::SP     userGeomsGroup;
    };
    std::vector<PerGPU> perGPU;
  };

  struct DataGroup : public Object {
    typedef std::shared_ptr<DataGroup> SP;

    DataGroup(Model *model, int localID);
    
    static SP create(Model *model, int localID)
    { return std::make_shared<DataGroup>(model,localID); }

    Group   *createGroup(const std::vector<Geom::SP> &geoms);
    Spheres *createSpheres(const mori::Material &material,
                           const vec3f *origins,
                           int numOrigins,
                           const float *radii,
                           float defaultRadius);
    
    struct {
      std::vector<Group::SP> groups;
      std::vector<affine3f>  xfms;
    } instances;

    void build();

    mori::DevGroup::SP const devGroup;
    Model *const model;
    int    const localID;
  };
  
  struct Model : public Object {
    typedef std::shared_ptr<Model> SP;

    static SP create(Context *ctx) { return std::make_shared<Model>(ctx); }
    
    Model(Context *context);
    
    /*! pretty-printer for printf-debugging */
    std::string toString() const override
    { return "Model{}"; }
    
    void build()
    {
      // todo: parallel
      for (auto &dg : dataGroups)
        dg->build();
    }
    void render(const mori::Camera *camera,
                FrameBuffer *fb);

    DataGroup *getDG(int localID)
    {
      assert(localID >= 0);
      assert(localID < dataGroups.size());
      return dataGroups[localID].get();
    }
    std::vector<DataGroup::SP> dataGroups;
    Context *const context;
  };

}
