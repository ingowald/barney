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

namespace tally_device {

struct Frame : public helium::BaseFrame
{
  Frame(TallyGlobalState *s);
  ~Frame() override;

  bool isValid() const override;

  TallyGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

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

  void convertPixelsToFinalFormat();

 private:
  void cleanup();

  bool m_valid{false};

  struct FrameData
  {
    int frameID{0};
    math::uint2 size;
    size_t totalPixels{0};
  } m_frameData;

  anari::DataType m_colorType{ANARI_UNKNOWN};
  anari::DataType m_depthType{ANARI_UNKNOWN};

  uint32_t *m_bnPixelBuffer{nullptr};
  uint8_t *m_colorBuffer{nullptr};
  float *m_depthBuffer{nullptr};

  helium::ChangeObserverPtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  mutable float m_duration{0.f};

  helium::TimeStamp m_cameraLastChanged{0};
  helium::TimeStamp m_rendererLastChanged{0};
  helium::TimeStamp m_worldLastChanged{0};
  helium::TimeStamp m_lastCommitOccured{0};
  helium::TimeStamp m_frameLastRendered{0};

  TallyFrame::SP m_bnFrameBuffer{nullptr};
};

} // namespace tally_device

TALLY_ANARI_TYPEFOR_SPECIALIZATION(tally_device::Frame *, ANARI_FRAME);
