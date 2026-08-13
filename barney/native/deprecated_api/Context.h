DEPRECATED

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/common/barney-common.h"
// #include "barney/api/common.h"
#include <mutex>
#include <set>

#ifndef BARNEY_NS
# error "BARNEY_NS not defined"
#endif

namespace BARNEY_NS {
  namespace barney_api {
    
  struct Context;
  struct Data;

  struct Renderer : public Object {
    Renderer(Context *context) : Object(context) {}
    virtual ~Renderer() = default;
  };

  struct Material : public Object {
    Material(Context *context) : Object(context) {}
    virtual ~Material() = default;
  };

  struct Geometry : public Object {
    Geometry(Context *context) : Object(context) {}
    virtual ~Geometry() = default;
  };

  struct Sampler : public Object {
    Sampler(Context *context) : Object(context) {}
    virtual ~Sampler() = default;
  };
 
  struct Light : public Object {
    Light(Context *context) : Object(context) {}
    virtual ~Light() = default;
  };
  
  struct Camera : public Object {
    Camera(Context *context) : Object(context) {}
    virtual ~Camera() = default;
  };
 
  
  struct Data : public Object {
    Data(Context *context) : Object(context) {}
    typedef std::shared_ptr<Data> SP;
    
    virtual ~Data() = default;
    
    virtual void set(const void *data, size_t count) = 0;
  };

  /*! object that handles a frame buffer object; in particular, the
    ability to let the renderer accumulate pixel samples, and to let
    the app 'read' different channels of the rendered frame buffer no
    matter on which gpu and/or rank they got written to */
  struct FrameBuffer : public Object {
    inline FrameBuffer(Context *context)
      : Object(context)
    {}
    virtual ~FrameBuffer() = default;

    virtual void  resetAccumulation() = 0;
    virtual void  resize(BNDataType colorFormat,
                         vec2i size,
                         uint32_t channels) = 0;
    virtual void  read(BNFrameBufferChannel channel,
                       void *hostPtrToReadInto,
                       BNDataType requestedFormat) = 0;
    /** Actual framebuffer dimensions (may differ from resize when e.g. upscaling forces even dims). */
    virtual vec2i getNumPixels() const { return vec2i(-1, -1); }
  };
  
  struct TextureData : public Object {
    TextureData(Context *context) : Object(context) {}
    virtual ~TextureData() = default;
  };

  struct ScalarField : public Object {
    ScalarField(Context *context) : Object(context) {}
    virtual ~ScalarField() = default;
  };

  struct Group : public Object {
    Group(Context *context) : Object(context) {}
    virtual ~Group() = default;
    virtual void build() = 0;
  };

  struct Volume : public Object {
    Volume(Context *context) : Object(context) {}
    virtual ~Volume() = default;
    virtual void setXF(const range1f &domain,
                       const bn_float4 *values,
                       int numValues,
                       float unitDensity) = 0;
  };
  
  struct Model : public Object{
    Model(Context *context) : Object(context) {}
    virtual void setInstances(int slot,
                              Group **groups,
                              const affine3f *xfms,
                              int numInstances) = 0;
    virtual void updateInstanceTransforms(int slot,
                                          const affine3f *xfms,
                                          int numInstances) = 0;
    virtual void setInstanceAttributes(int slot,
                                       const std::string &which,
                                       Data::SP data) = 0;
    virtual void build(int slot) = 0;
    virtual void render(Renderer *renderer,
                        Camera *camera,
                        FrameBuffer *fb) = 0;
  };
  
  struct Texture : public Object {
    Texture(Context *context) : Object(context) {}
    virtual ~Texture() = default;
  };


  struct LocalSlot {
    int dataRank;
    std::vector<int> gpuIDs;
  };
  
  struct Context {
    Context(const std::vector<LocalSlot> &localSlot) {};
    virtual ~Context() = default;

    virtual int myRank() = 0;
    virtual int mySize() = 0;
    
    // ------------------------------------------------------------------
    // virtual object factory interface
    // ------------------------------------------------------------------
    
    // ----------- non-slotted object types -----------
    
    virtual std::shared_ptr<Model>
    createModel() = 0;
    
    virtual std::shared_ptr<Renderer>
    createRenderer() = 0;

    virtual std::shared_ptr<Camera>
    createCamera(const std::string &type) = 0;

    virtual std::shared_ptr<FrameBuffer>
    createFrameBuffer() = 0;

    // ----------- slotted object types -----------
    
    virtual std::shared_ptr<TextureData> 
    createTextureData(int slot,
                      BNDataType texelFormat,
                      vec3i dims,
                      const void *texels) = 0;
    
    virtual std::shared_ptr<ScalarField>
    createScalarField(int slot, const std::string &type) = 0;
    
    virtual std::shared_ptr<Geometry>
    createGeometry(int slot, const std::string &type) = 0;
    
    virtual std::shared_ptr<Material>
    createMaterial(int slot, const std::string &type) = 0;

    virtual std::shared_ptr<Sampler>
    createSampler(int slot, const std::string &type) = 0;

    virtual std::shared_ptr<Light>
    createLight(int slot, const std::string &type) = 0;

    virtual std::shared_ptr<Group>
    createGroup(int slot,
                Geometry **geoms, int numGeoms,
                Volume **volumes, int numVolumes) = 0;

    virtual std::shared_ptr<Data>
    createData(int slot,
               BNDataType dataType) = 0;
    
    // ----------- implicitly slotted object types -----------
    
    virtual std::shared_ptr<Texture>
    createTexture(const std::shared_ptr<TextureData> &td,
                  BNTextureFilterMode  filterMode,
                  BNTextureAddressMode addressModes[],
                  BNTextureColorSpace  colorSpace = BN_COLOR_SPACE_LINEAR) = 0;

    virtual std::shared_ptr<Volume>
    createVolume(const std::shared_ptr<ScalarField> &sf) = 0;

    

  };

  /*! pretty-printer for printf-debugging */
  inline std::string Object::toString() const
  { return "<Object>"; }

  /*! dynamically cast to another (typically derived) class, e.g. to
    check whether a given 'Geomery'-type object is actually a
    Triangles-type geometry, etc */
  template<typename T>
  inline std::shared_ptr<T> Object::as() 
  { return std::dynamic_pointer_cast<T>(shared_from_this()); }

  inline void Object::warn_unsupported_member(const std::string &type,
                                              const std::string &member)
  {
    static std::set<std::string> alreadyWarned;
    std::string key = toString()+"_"+type+"_"+member;
    if (// context->
        alreadyWarned.find(key) != // context->
        alreadyWarned.end())
      return;
    std::cout << OWL_TERMINAL_RED
              << "#bn: warning - invalid member access. "
              << "Object '" << toString() << "' does not have a member '"<<member<<"'"
              << " of type '"<< type << "'"
              << OWL_TERMINAL_DEFAULT << std::endl;
    // context->
      alreadyWarned.insert(key);
  }

  inline void Context::releaseHostReference(Object::SP object)
  {
    auto it = hostOwnedHandles.find(object);
    if (it == hostOwnedHandles.end())
      throw std::runtime_error
        ("trying to bnRelease() a handle that either does not "
         "exist, or that the app (no lnoger) has any valid references on");

    const int remainingReferences = --it->second;

    if (remainingReferences == 0) {
      // remove the std::shared-ptr handle:
      it->second = {};
      // and make barney forget that it ever had this object 
      hostOwnedHandles.erase(it);
    }
  }
  
  inline void Context::addHostReference(Object::SP object)
  {
    auto it = hostOwnedHandles.find(object);
    if (it == hostOwnedHandles.end())
      throw std::runtime_error
        ("trying to bnAddReference() to a handle that either does not "
         "exist, or that the app (no lnoger) has any valid primary references on");
    
    // add one ref count:
    it->second++;
  }
  }
  
  struct FromEnv {
    FromEnv();
    static const FromEnv *get();

    static bool enabled(const std::string &key)
    {
      auto &boolValues = get()->boolValues;
      auto it = boolValues.find(key);
      if (it == boolValues.end()) return false;
      return it->second;
    }
    /*! allows for querying whether a value _was_ set _and_ set to
        false. E.g, 'denoising=0' will return true for
        explicitDisabled("denosing"); "denoising=1' would return false
        (because it's _en_abled, not disabled), and 'denoising' not
        set at all would return false (because it hasn't even been
        set, and thus not explicitly disabled */
    static bool explicitlyDisabled(const std::string &key)
    {
      auto &boolValues = get()->boolValues;
      auto it = boolValues.find(key);
      if (it == boolValues.end()) return false;
      return !it->second;
    }
    
    std::map<std::string,bool> boolValues;
    
    bool logQueues  = false;
    bool skipDenoising = false;
    bool logConfig  = false;
    bool logBackend = false;
    bool logTopo    = false;
  };

}
