// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"
#include "Renderer.h"
#include "World.h"
// helium
#include "helium/BaseFrame.h"
// std
#include <vector>

namespace barney_device {

struct Frame : public helium::BaseFrame
{
  Frame(BarneyGlobalState *s);
  ~Frame() override;

  bool isValid() const override;

  BarneyGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  void renderFrame() override;

  void *map(std::string_view channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;

  bool ready() const;
  void wait() const;

 private:
  void cleanup();

  bool m_valid{false};

  struct FrameData
  {
    int frameID{0};
    math::uint2 size;
  } m_frameData;

  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};

  uint32_t *m_colorBuffer{nullptr};
  float    *m_depthBuffer{nullptr};

  helium::ChangeObserverPtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;
  helium::TimeStamp m_lastCommitFlush{0};

  mutable float m_duration{0.f};

  BNFrameBuffer m_bnFrameBuffer{nullptr};
};

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Frame *, ANARI_FRAME);
