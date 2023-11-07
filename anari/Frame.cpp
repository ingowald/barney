// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// std
#include <chrono>

namespace barney_device {

Frame::Frame(BarneyGlobalState *s) : helium::BaseFrame(s)
{
  s->objectCounts.frames++;

  m_bnFrameBuffer = bnFrameBufferCreate(s->context, 0);
}

Frame::~Frame()
{
  wait();
  deviceState()->objectCounts.frames--;
}

bool Frame::isValid() const
{
  return m_valid;
}

BarneyGlobalState *Frame::deviceState() const
{
  return (BarneyGlobalState *)helium::BaseObject::m_state;
}

void Frame::commit()
{
  m_renderer = getParamObject<Renderer>("renderer");
  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  m_camera = getParamObject<Camera>("camera");
  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  m_world = getParamObject<World>("world");
  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  m_valid = m_renderer && m_renderer->isValid() && m_camera
      && m_camera->isValid() && m_world && m_world->isValid();

  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);

  auto size = getParam<uint2>("size", make_uint2(10, 10));

  const auto numPixels = size.x * size.y;

  auto perPixelBytes = 4 * (m_colorType == ANARI_FLOAT32_VEC4 ? 4 : 1);
  m_pixelBuffer.resize(numPixels * perPixelBytes);

  m_depthBuffer.resize(m_depthType == ANARI_FLOAT32 ? numPixels : 0, 0.f);

  m_bnHostBuffer.resize(numPixels);
  bnFrameBufferResize(m_bnFrameBuffer,
      size.x,
      size.y,
      m_bnHostBuffer.data(),
      nullptr);
  m_frameData.size = size;
}

bool Frame::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    if (flags & ANARI_WAIT)
      wait();
    helium::writeToVoidP(ptr, m_duration);
    return true;
  }

  return 0;
}

void Frame::renderFrame()
{
  auto *state = deviceState();
  state->waitOnCurrentFrame();

  auto start = std::chrono::steady_clock::now();

  state->commitBuffer.flush();

  if (!isValid()) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
    return;
  }

  if (state->commitBuffer.lastFlush() > m_frameLastRendered)
    bnAccumReset(m_bnFrameBuffer);

  m_world->barneyModelUpdate();

  m_frameLastRendered = helium::newTimeStamp();
  state->currentFrame = this;

  bnRender(m_world->barneyModel(), m_camera->barneyCamera(), m_bnFrameBuffer);

  auto end = std::chrono::steady_clock::now();
  m_duration = std::chrono::duration<float>(end - start).count();
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  wait();

  *width = m_frameData.size.x;
  *height = m_frameData.size.y;

  if (channel == "channel.color") {
    *pixelType = m_colorType;
    return m_colorType == ANARI_UFIXED8_VEC4 ? (void *)m_bnHostBuffer.data()
                                             : (void *)m_pixelBuffer.data();
  } else if (channel == "channel.depth" && !m_depthBuffer.empty()) {
    *pixelType = ANARI_FLOAT32;
    return m_depthBuffer.data();
  }

  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
}

void Frame::unmap(std::string_view channel)
{
  // no-op
}

int Frame::frameReady(ANARIWaitMask m)
{
  if (m == ANARI_NO_WAIT)
    return ready();
  else {
    wait();
    return 1;
  }
}

void Frame::discard()
{
  // no-op
}

bool Frame::ready() const
{
  return true; // not yet async
}

void Frame::wait() const
{
  deviceState()->currentFrame = nullptr;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Frame *);
