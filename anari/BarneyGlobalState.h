// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_math.h"
// helium
#include "helium/BaseGlobalDeviceState.h"

namespace barney_device {

struct Frame;

struct BarneyGlobalState : public helium::BaseGlobalDeviceState
{
  struct ObjectCounts
  {
    size_t frames{0};
    size_t cameras{0};
    size_t renderers{0};
    size_t worlds{0};
    size_t instances{0};
    size_t groups{0};
    size_t lights{0};
    size_t surfaces{0};
    size_t geometries{0};
    size_t materials{0};
    size_t samplers{0};
    size_t volumes{0};
    size_t spatialFields{0};
    size_t arrays{0};
    size_t unknown{0};
  } objectCounts;

  struct ObjectUpdates
  {
    helium::TimeStamp lastBLSReconstructSceneRequest{0};
    helium::TimeStamp lastBLSCommitSceneRequest{0};
    helium::TimeStamp lastTLSReconstructSceneRequest{0};
  } objectUpdates;

  Frame *currentFrame{nullptr};

  BNContext context{nullptr};

  bool allowInvalidSurfaceMaterials{true};
  float4 invalidMaterialColor{1.f, 0.f, 1.f, 1.f};

  // Helper methods //

  BarneyGlobalState(ANARIDevice d);
  void waitOnCurrentFrame() const;
};

// Helper functions/macros ////////////////////////////////////////////////////

inline BarneyGlobalState *asBarneyState(helium::BaseGlobalDeviceState *s)
{
  return (BarneyGlobalState *)s;
}

#define BARNEY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);                              \
  }

#define BARNEY_ANARI_TYPEFOR_DEFINITION(type)                                  \
  namespace anari {                                                            \
  ANARI_TYPEFOR_DEFINITION(type);                                              \
  }

} // namespace barney_device
