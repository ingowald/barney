// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/common/barney-common.h"
// #include "api/Context.h"
#include "barney/DeviceGroup.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Context;
    struct Data;
    struct ModelSlot;
    struct World;

    /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
    struct Object : public std::enable_shared_from_this<Object> {
      typedef std::shared_ptr<Object> SP;

      Object(Context *context);
      virtual ~Object();

      /*! dynamically cast to another (typically derived) class, e.g. to
        check whether a given 'Geomery'-type object is actually a
        Triangles-type geometry, etc */
      template<typename T>
      inline std::shared_ptr<T> as();
      template<typename T>
      inline std::shared_ptr<const T> as() const;
    
      /*! pretty-printer for printf-debugging */
      virtual std::string toString() const;

      void warn_unsupported_member(const std::string &type,
                                   const std::string &member);

      // ------------------------------------------------------------------
      /*! @{ parameter set/commit interface */
      virtual void commit() {}
      virtual bool setObject(const std::string &member,
                             const Object::SP &value)
      { return false; }
      virtual bool setData(const std::string &member,
                           const std::shared_ptr<Data> &value)
      { return false; }
      virtual bool setString(const std::string &member,
                             const std::string &value)
      { return false; }
      virtual bool set1f(const std::string &member, const float &value) { return false; }
      virtual bool set2f(const std::string &member, const vec2f &value) { return false; }
      virtual bool set3f(const std::string &member, const vec3f &value) { return false; }
      virtual bool set4f(const std::string &member, const vec4f &value) { return false; }
      virtual bool set1i(const std::string &member, const int   &value) { return false; }
      virtual bool set2i(const std::string &member, const vec2i &value) { return false; }
      virtual bool set3i(const std::string &member, const vec3i &value) { return false; }
      virtual bool set4i(const std::string &member, const vec4i &value) { return false; }
      virtual bool set4x3f(const std::string &member, const affine3f &value) { return false; }
      virtual bool set4x4f(const std::string &member, const vec4f *value) { return false; }
      /*! @} */
      // ------------------------------------------------------------------

      /*! returns the context that this object was created in */
      Context *getContext() const { return context; }
    
      // NOT a shared pointer to avoid cyclical dependencies.
      Context *const context;
    };
    
    /*! a object owned (only) in a particular data group */
    struct SlottedObject : public Object {
      SlottedObject(Context *context, const DevGroup::SP &devices);
      virtual ~SlottedObject() = default;

      /*! pretty-printer for printf-debugging */
      std::string toString() const override { return "<SlottedObject>"; }

      const DevGroup::SP devices;    
    };

    struct Data : public Object {
      Data(Context *context) : Object(context) {}
      typedef std::shared_ptr<Data> SP;
    
      virtual ~Data() = default;
    
      virtual void set(const void *data, size_t count) = 0;
    };

    
    // ==================================================================
    // INLINE IMPLEMENTATION SECTION
    // ==================================================================
    
    /*! dynamically cast to another (typically derived) class, e.g. to
      check whether a given 'Geomery'-type object is actually a
      Triangles-type geometry, etc */
    template<typename T>
    inline std::shared_ptr<T> Object::as() 
    { return std::dynamic_pointer_cast<T>(shared_from_this()); }

  }
}
