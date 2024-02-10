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

#include "barney/Data.h"

namespace barney {

  struct Context;
  
  /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;

    Object(Context *context) : context(context) {}
    virtual ~Object() {}

    /*! dynamically cast to another (typically derived) class, e.g. to
        check whether a given 'Geomery'-type object is actually a
        Triangles-type geometry, etc */
    template<typename T>
    inline std::shared_ptr<T> as();
    template<typename T>
    inline std::shared_ptr<const T> as() const;
    
    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const;


    virtual bool set(const std::string &arg, const std::shared_ptr<Object> &value)
    { return false; }
    virtual bool set(const std::string &arg, const Data::SP &value)
    { return false; }
    virtual bool set(const std::string &arg, const int    value) { return false; }
    virtual bool set(const std::string &arg, const float &value) { return false; }
    virtual bool set(const std::string &arg, const vec2f &value) { return false; }
    virtual bool set(const std::string &arg, const vec3f &value) { return false; }
    virtual bool set(const std::string &arg, const vec4f &value) { return false; }
    virtual bool set(const std::string &arg, const int    value) { return false; }
    virtual bool set(const std::string &arg, const vec2i &value) { return false; }
    virtual bool set(const std::string &arg, const vec3i &value) { return false; }
    virtual bool set(const std::string &arg, const vec4i &value) { return false; }
    
    virtual void commit() {}

    
    /*! returns the context that this object was created in */
    Context *getContext() const { return context; }
    
    // NOT a shared pointer to avoid cyclical dependencies.
    Context *const context;
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
