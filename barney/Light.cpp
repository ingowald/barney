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

#include "barney/Light.h"
#include "barney/DataGroup.h"
#include "barney/Context.h"

namespace barney {

  Light::SP Light::create(DataGroup *owner,
                          const std::string &type)
  {
    if (type == "directional")
      return std::make_shared<DirLight>(owner);
    if (type == "quad")
      return std::make_shared<QuadLight>(owner);
    if (type == "environment")
      return std::make_shared<EnvMapLight>(owner);
    
    owner->context->warn_unsupported_object("Light",type);
    return {};
  }

  // ==================================================================
  
  bool DirLight::set3f(const std::string &member, const vec3f &value) 
  {
    if (member == "direction") {
      content.direction = value;
      return true;
    }
    if (member == "radiance") {
      content.radiance = value;
      return true;
    }
    return 0;
  }

  // ==================================================================
  
  bool QuadLight::set3f(const std::string &member, const vec3f &value) 
  {
    return 0;
  }

  // ==================================================================
  
  void EnvMapLight::commit()
  {
    if (envMap.texels) {
      envMap.texelsBuffer = envMap.texels->owl;
      std::vector<vec4f> texels(envMap.texels->count);
      SetActiveGPU forDuration(owner->devGroup->devices[0]);
      BARNEY_CUDA_CALL(Memcpy(texels.data(),owlBufferGetPointer(envMap.texels->owl,0),
                              envMap.texels->count*sizeof(vec4f),cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      float sum = 0.f;
      for (int i=0;i<envMap.texels->count;i++) {
        sum += reduce_max((const vec3f&)texels[i]);
        texels[i].w = sum;
      }
      float rcp_sum = 1.f/sum;
      for (int i=0;i<envMap.texels->count;i++) 
        texels[i].w *= rcp_sum;
      std::cout << "done computing CDF of envmap..." << std::endl;
      owlBufferUpload(envMap.texels->owl,texels.data());
    }
  }
  
  bool EnvMapLight::set2i(const std::string &member, const vec2i &value) 
  {
    if (member == "envMap.dims") {
      envMap.dims = value;
      return true;
    }
    return 0;
  }

  bool EnvMapLight::set3f(const std::string &member, const vec3f &value) 
  {
    return 0;
  }

  bool EnvMapLight::set4x3f(const std::string &member, const affine3f &value) 
  {
    if (member == "envMap.transform") {
      envMap.transform = value;
      return true;
    }
    return 0;
  }

  bool EnvMapLight::setData(const std::string &member, const Data::SP &value) 
  {
    if (member == "envMap.texels") {
      envMap.texels = value->as<PODData>();
      return true;
    }
    return 0;
  }

};
