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

#include "barney.h"
#include <string.h>
#include <cuda_runtime.h>
#include <mutex>
#include <map>

namespace barney {

  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;

    template<typename T>
    inline std::shared_ptr<T> as()
    { return std::dynamic_pointer_cast<T>(shared_from_this()); }
    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const
    { return "<Object>"; }
  };

}
