// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney_math.h"
// helium
#include "helium/BaseGlobalDeviceState.h"
#include <memory>
#include <map>

namespace barney_device {

  inline void bnSetAndRelease(BNObject o, const char *n, BNObject v)
  {
    bnSetObject(o,n,v);
    bnRelease(v);
  }
  inline void bnSetAndRelease(BNObject o, const char *n, BNData v)
  {
    bnSetData(o,n,v);
    bnRelease(v);
  }

  struct Frame;
  struct World;

  struct BarneyDevice;

  struct TetheredModel {
    BNModel model;
  };

  /*! keeps info on multiple (banari-)devices that are tethered
      together onto a singel barney ncontext */
  struct Tether {
    BNContext context{nullptr};

    bool allDevicesPresent();

    TetheredModel *getModel(int uniqueID);
    void releaseModel(int uniqueID);
    std::map<int,std::pair<int,std::shared_ptr<TetheredModel>>> activeModels;
    std::mutex mutex;

    int numRenderCallsOutstanding = 0;
    struct {
      BNCamera camera;
      BNRenderer renderer;
      BNFrameBuffer fb;
      BNModel model;
    } deferredRenderCall;

    std::vector<BarneyDevice *> devices;
  };

  struct BarneyGlobalState : public helium::BaseGlobalDeviceState
  {
    struct ObjectUpdates
    {
      helium::TimeStamp lastSceneChange{0};
    } objectUpdates;

    int slot = -1;

    std::shared_ptr<Tether> tether;

    /*! created models get consecutive IDs, which allows us for
        identifying which models created by which (tethered) device(s)
        belong togther. Ie, if two devices A and B are tethered, then
        the i'th model of A is always tethered with the i'th model of
        B (and vice versa) */
    int nextUniqueModelID = 0;

    bool hasBeenCommitted = false;

    // Helper methods //

    BarneyGlobalState(ANARIDevice d);
    void markSceneChanged();
  };

  // Helper functions/macros ////////////////////////////////////////////////////

  inline BarneyGlobalState *asBarneyState(helium::BaseGlobalDeviceState *s)
  {
    return (BarneyGlobalState *)s;
  }

#define BARNEY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)   \
  namespace anari {                                             \
    ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);             \
  }

#define BARNEY_ANARI_TYPEFOR_DEFINITION(type)   \
  namespace anari {                             \
    ANARI_TYPEFOR_DEFINITION(type);             \
  }

} // namespace barney_device
