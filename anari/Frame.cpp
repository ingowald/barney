// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// std
#include <algorithm>
#include <chrono>

namespace barney_device {

Frame::Frame(BarneyGlobalState *s) : helium::BaseFrame(s)
{
  m_bnFrameBuffer = bnFrameBufferCreate(s->context, 0);
}

Frame::~Frame()
{
  wait();
  bnRelease(m_bnFrameBuffer);
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

  auto size = getParam<math::uint2>("size", math::uint2(10, 10));

  const auto numPixels = size.x * size.y;

  auto perPixelBytes = 4 * (m_colorType == ANARI_FLOAT32_VEC4 ? 4 : 1);
  m_pixelBuffer.resize(numPixels * perPixelBytes);

  m_depthBuffer.resize(m_depthType == ANARI_FLOAT32 ? numPixels : 0, 0.f);

  m_bnHostBuffer.resize(numPixels);
  bnFrameBufferResize(m_bnFrameBuffer,
      size.x,
      size.y,
      m_bnHostBuffer.data(),
      m_depthBuffer.data());
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

  state->commitBufferFlush();

  if (!isValid()) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
    return;
  }

  if (state->commitBufferLastFlush() > m_frameLastRendered)
    bnAccumReset(m_bnFrameBuffer);

  m_world->barneyModelUpdate();

  m_frameLastRendered = helium::newTimeStamp();
  state->currentFrame = this;

  const int pixelSamples = std::max(m_renderer->pixelSamples(), 1);

  for (int i = 0; i < pixelSamples; i++)
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
    convertPixelsToFinalFormat();
    *pixelType = m_colorType;
    return m_pixelBuffer.data();
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

void Frame::convertPixelsToFinalFormat()
{
  if (m_colorType == ANARI_UFIXED8_VEC4) {
    std::memcpy(
        m_pixelBuffer.data(), m_bnHostBuffer.data(), m_pixelBuffer.size());
  } else if (m_colorType == ANARI_UFIXED8_RGBA_SRGB) {
    auto numPixels = m_frameData.size.x * m_frameData.size.y;
    auto *src = (math::byte4 *)m_bnHostBuffer.data();
    auto *dst = (math::byte4 *)m_pixelBuffer.data();
    std::transform(src, src + numPixels, dst, [](math::byte4 p) {
      auto f = math::float4(p.x / 255.f, p.y / 255.f, p.z / 255.f, p.w / 255.f);
      f.x = helium::toneMap(f.x);
      f.y = helium::toneMap(f.y);
      f.z = helium::toneMap(f.z);
      return math::byte4(f.x * 255, f.y * 255, f.z * 255, f.w * 255);
    });
  } else if (m_colorType == ANARI_FLOAT32_VEC4) {
    auto numPixels = m_frameData.size.x * m_frameData.size.y;
    auto *src = (math::byte4 *)m_bnHostBuffer.data();
    auto *dst = (math::float4 *)m_pixelBuffer.data();
    std::transform(src, src + numPixels, dst, [](math::byte4 p) {
      return math::float4(p.x / 255.f, p.y / 255.f, p.z / 255.f, p.w / 255.f);
    });
  }
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Frame *);
