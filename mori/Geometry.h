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

#include "mori/DeviceGroup.h"

namespace mori {

  struct Material {
    vec3f diffuseColor;
  };
  
  struct Geometry {
    typedef std::shared_ptr<Geometry> SP;

    Geometry(DevGroup *devGroup,
         const Material &material)
      : devGroup(devGroup),
        material(material)
    {}

    Material     material;
    DevGroup    *devGroup  = 0;
  };

  struct Group {
    typedef std::shared_ptr<Group> SP;

    Group(DevGroup *devGroup)
      : devGroup(devGroup)
    {}
    
    virtual void build() = 0;

    DevGroup         *devGroup  = 0;
    std::vector<OWLGroup> perDev;
  };
  
  struct TriangleGeomsGroup : public Group {
    TriangleGeomsGroup(DevGroup *devGroup,
                       const std::vector<Geometry::SP> &triangleGeoms)
      : Group(devGroup),
        triangleGeoms(triangleGeoms)
    {}
    
    void build() override { PING; }
    std::vector<Geometry::SP> triangleGeoms;
  };
  
  struct UserGeomsGroup : public Group {
    UserGeomsGroup(DevGroup *devGroup,
                   const std::vector<Geometry::SP> &userGeoms)
      : Group(devGroup),
        userGeoms(userGeoms)
    {}
    
    void build() override { PING; }
    const std::vector<Geometry::SP> userGeoms;
  };
  struct InstanceGroup : public Group {
    InstanceGroup(DevGroup *devGroup,
                  const std::vector<Group::SP> &groups,
                  const std::vector<affine3f>  &xfms)
      : Group(devGroup),
        groups(groups),
        xfms(xfms)
    {}
        
    void build() override { PING; }
    
    const std::vector<Group::SP> groups;
    const std::vector<affine3f>  xfms;
  };
}
