// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_math.h"
// helium
#include "helium/BaseGlobalDeviceState.h"

namespace barney_device {

struct Frame;
struct World;

struct BarneyGlobalState : public helium::BaseGlobalDeviceState
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

  BarneyGlobalState(ANARIDevice d);
  void waitOnCurrentFrame() const;
  void markSceneChanged();
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
