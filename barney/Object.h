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

#include "barney/common/barney-common.h"

namespace barney {

  /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;

    /*! dynamically cast to another (typically derived) class, e.g. to
        check whether a given 'Geomery'-type object is actually a
        Triangles-type geometry, etc */
    template<typename T>
    inline std::shared_ptr<T> as();
    template<typename T>
    inline std::shared_ptr<const T> as() const;
    
    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const;
  };


  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================
  
  /*! pretty-printer for printf-debugging */
  inline std::string Object::toString() const
  { return "<Object>"; }

  /*! dynamically cast to another (typically derived) class, e.g. to
    check whether a given 'Geomery'-type object is actually a
    Triangles-type geometry, etc */
  template<typename T>
  inline std::shared_ptr<T> Object::as() 
  { return std::dynamic_pointer_cast<T>(shared_from_this()); }
  
}
