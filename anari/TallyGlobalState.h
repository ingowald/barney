// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tally_math.h"
// helium
#include "helium/BaseGlobalDeviceState.h"
#include <memory>
#include <vector>
#include <iostream>

namespace tally_device {

  struct StructuredRegularField;
  struct UnstructuredField;
  struct Surface;  
  struct TallyCamera {
    typedef std::shared_ptr<TallyCamera> SP;
    static SP create(const std::string &type) { return std::make_shared<TallyCamera>(); }
  };

  struct TallyMaterial {
    typedef std::shared_ptr<TallyMaterial> SP;
    static SP create(const std::string &type) { return std::make_shared<TallyMaterial>(); }
  };

  struct TallySampler {
    typedef std::shared_ptr<TallySampler> SP;
    static SP create(const std::string &type) { return std::make_shared<TallySampler>(); }
  };

  struct TallyGeom {
    virtual ~TallyGeom() {}
    typedef std::shared_ptr<TallyGeom> SP;
    static SP create(const std::string &type, const Surface *surf);// { return std::make_shared<TallyGeom>(); }
    TallyMaterial::SP material;
  };

  struct TallyTrianglesGeom : public TallyGeom {
    typedef std::shared_ptr<TallyTrianglesGeom> SP;
    TallyTrianglesGeom(const Surface *surface) : surface(surface) {
      for (int i=0;i<1+random()%5; i++)
        chunks.push_back(Chunk{});
    }

    struct Chunk { /* whatever */ int foo; };
    std::vector<Chunk> chunks;
    const Surface *surface;
  };

  inline TallyGeom::SP TallyGeom::create(const std::string &type, const Surface *surf)
  { //return std::make_shared<TallyGeom>();
    std::cout << "chunking geom of " << type << std::endl;
    if (type == "triangles")
      return std::make_shared<TallyTrianglesGeom>(surf);
    throw std::runtime_error("can only do triangles...");
  }

  struct TallyLight {
    typedef std::shared_ptr<TallyLight> SP;
    static SP create(const std::string &type) { return std::make_shared<TallyLight>(); }
  };

  struct TallyFrame {
    typedef std::shared_ptr<TallyFrame> SP;
    static SP create() { return std::make_shared<TallyFrame>(); }
  };

  // struct TallySurface {
  //   typedef std::shared_ptr<TallySurface> SP;
  //   static SP create() { return std::make_shared<TallySurface>(); }
  // };

  struct TallySpatialField {
    typedef std::shared_ptr<TallySpatialField> SP;
    static SP create(const std::string &type) { return std::make_shared<TallySpatialField>(); }
  };
  struct TallyVolume {
    typedef std::shared_ptr<TallyVolume> SP;
    virtual void update() {}
    // static SP create() { return std::make_shared<TallyVolume>(); }
  };
  struct TallyStructuredData : public TallySpatialField {
    typedef std::shared_ptr<TallyStructuredData> SP;
    static SP create(const StructuredRegularField *frontend) { return std::make_shared<TallyStructuredData>(); }
  };
  struct TallyUMeshField : public TallySpatialField {
    typedef std::shared_ptr<TallyUMeshField> SP;
    static SP create(const UnstructuredField *frontend) { return std::make_shared<TallyUMeshField>(); }
  };

  struct TallyTransferFunction1D : public TallyVolume {
    typedef std::shared_ptr<TallyTransferFunction1D> SP;
    TallyTransferFunction1D(TallySpatialField::SP field) : field(field) {}
    static SP create(TallySpatialField::SP field) { return std::make_shared<TallyTransferFunction1D>(field); }
    TallySpatialField::SP field;
  };

  
  struct TallyGroup {
    typedef std::shared_ptr<TallyGroup> SP;

    TallyGroup(const std::vector<TallyGeom::SP>   &geoms,
               const std::vector<TallyVolume::SP> &volumes,
               const std::vector<TallyLight::SP>  &lights)
      : geoms(geoms), volumes(volumes), lights(lights)
    {}

    void print()
    {
      std::cout << "Group " << (int*)this << std::endl;
      std::cout << "  #geoms " << geoms.size() << std::endl;
      std::cout << "  #volumes " << volumes.size() << std::endl;
    }
    static SP create(const std::vector<TallyGeom::SP>   &geoms,
                     const std::vector<TallyVolume::SP> &volumes,
                     const std::vector<TallyLight::SP>  &lights)
    {
      return std::make_shared<TallyGroup>(geoms,volumes,lights);
    }
                     
    std::vector<TallyGeom::SP>   geoms;
    std::vector<TallyVolume::SP> volumes;
    std::vector<TallyLight::SP>  lights;
  };

  struct TallyTransform {
    math::mat4 xfm;
  };
  struct TallyModel {
    typedef std::shared_ptr<TallyModel> SP;
    static SP create() { return std::make_shared<TallyModel>(); }
    void setRadiance(float)
    {}
    void render(TallyCamera::SP camera, TallyFrame::SP frame, int pixelSamples)
    {}
    void setInstances(const std::vector<TallyGroup::SP> &groups,
                      const std::vector<TallyTransform> &transforms)
    {
      std::cout << "----- setting instances: " << std::endl;
      std::cout << " -- groups" << std::endl;
      for (auto &group : groups) {
        if (group)
          group->print();
        for (auto geom : group->geoms) {
          TallyTrianglesGeom::SP tris = std::dynamic_pointer_cast<TallyTrianglesGeom>(geom);
          if (tris) {
            for (auto &chunk : tris->chunks)
              std::cout << "  -> have chunk " << &chunk << std::endl;
          }
        }
      }
      std::cout << " -- instances thereof" << std::endl;
      for (int i=0;i<groups.size();i++) 
        std::cout << "   #" << i << " : " << "<xfm> x " << groups[i].get() << std::endl;
      this->groups = groups;
      this->transforms = transforms;
    }
    std::vector<TallyGroup::SP> groups;
    std::vector<TallyTransform> transforms;
  };
  
  
struct Frame;
struct World;

struct TallyGlobalState : public helium::BaseGlobalDeviceState
{
  struct ObjectUpdates
  {
    helium::TimeStamp lastSceneChange{0};
  } objectUpdates;

  Frame *currentFrame{nullptr};
  World *currentWorld{nullptr};

  BNContext context{nullptr};

  bool allowInvalidSurfaceMaterials{true};
  math::float4 invalidMaterialColor{1.f, 0.f, 1.f, 1.f};

  BNHardwareInfo bnInfo;

  // Helper methods //

  TallyGlobalState(ANARIDevice d);
  void waitOnCurrentFrame() const;
  void markSceneChanged();
};

// Helper functions/macros ////////////////////////////////////////////////////

inline TallyGlobalState *asTallyState(helium::BaseGlobalDeviceState *s)
{
  return (TallyGlobalState *)s;
}

#define TALLY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define TALLY_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // namespace tally_device
