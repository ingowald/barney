// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
#if 1
// std
#include <algorithm>
#include <chrono>
#include <iostream>
// cuda
// // thrust
// #include <thrust/execution_policy.h>
// #include <thrust/transform.h>

#ifndef PRINT
#define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
#define PING                                                                   \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__            \
            << std::endl;
#else
#define PING                                                                   \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__     \
            << std::endl;
#endif
#endif

namespace barney_device {

Frame::Frame(BarneyGlobalState *s) : helium::BaseFrame(s), m_renderer(this)
{
  m_bnFrameBuffer = bnFrameBufferCreate(s->context, 0);
}

Frame::~Frame()
{
  wait();
  cleanup();
  bnRelease(m_bnFrameBuffer);
}

bool Frame::isValid() const
{
  return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid();
}

BarneyGlobalState *Frame::deviceState() const
{
  return (BarneyGlobalState *)helium::BaseObject::m_state;
}

BNDataType toBarney(anari::DataType type)
{
  switch (type) {
  case ANARI_UFIXED8_VEC4:
    return BN_UFIXED8_RGBA;
  case ANARI_UFIXED8_RGBA_SRGB:
    return BN_UFIXED8_RGBA_SRGB;
  case ANARI_FLOAT32:
    return BN_FLOAT;
  case ANARI_FLOAT32_VEC3:
    return BN_FLOAT3;
  case ANARI_FLOAT32_VEC4:
    return BN_FLOAT4;
  }
  throw std::runtime_error("toBarney: anari data type %i not handled yet");
}

void Frame::commitParameters()
{
  m_renderer = getParamObject<Renderer>("renderer");
  m_camera = getParamObject<Camera>("camera");
  m_world = getParamObject<World>("world");
  m_colorType = getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_depthType = getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
  m_frameData.size = getParam<math::uint2>("size", math::uint2(10, 10));
}

void Frame::finalize()
{
  cleanup();

  if (!m_renderer) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  }

  if (!m_camera) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  }

  if (!m_world) {
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  }

  const auto &size = m_frameData.size;
  const auto numPixels = size.x * size.y;

  m_colorBuffer =
      new uint32_t[numPixels * (m_colorType == ANARI_FLOAT32_VEC4 ? 4 : 1)];
  if (m_depthType == ANARI_FLOAT32)
    m_depthBuffer = new float[numPixels];

  bnFrameBufferResize(m_bnFrameBuffer,
      size.x,
      size.y,
      (uint32_t)BN_FB_COLOR
          | (uint32_t)((m_depthType == ANARI_FLOAT32) ? BN_FB_DEPTH : 0));
  bnSet1i(m_bnFrameBuffer,
      "showCrosshairs",
      m_renderer ? int(m_renderer->crosshairs()) : false);
  bnCommit(m_bnFrameBuffer);
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
  auto start = std::chrono::steady_clock::now();

  auto *state = deviceState();
  state->commitBuffer.flush();

  if (m_lastCommitFlush < state->commitBuffer.lastFlush()) {
    m_lastCommitFlush = helium::newTimeStamp();
    bnAccumReset(m_bnFrameBuffer);
  }

  if (!isValid()) {
    reportMessage(
        ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
    reportMessage(ANARI_SEVERITY_DEBUG,
        "    renderer(%p) - isValid:(%i)",
        m_renderer.get(),
        m_renderer ? m_renderer->isValid() : 0);
    reportMessage(ANARI_SEVERITY_DEBUG,
        "    world(%p) - isValid:(%i)",
        m_world.ptr,
        m_world ? m_world->isValid() : 0);
    reportMessage(ANARI_SEVERITY_DEBUG,
        "    camera(%p) - isValid:(%i)",
        m_camera.ptr,
        m_camera ? m_camera->isValid() : 0);
    return;
  }

  auto model = m_world->makeCurrent();

  bnRender(m_renderer->barneyRenderer,
      model,
      m_camera->barneyCamera(),
      m_bnFrameBuffer);

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
    bnFrameBufferRead(m_bnFrameBuffer, BN_FB_COLOR,
                      m_colorBuffer, toBarney(m_colorType));
    *pixelType = m_colorType;
    return m_colorBuffer;
  } else if (channel == "channel.depth" && m_depthBuffer) {
    bnFrameBufferRead(m_bnFrameBuffer, BN_FB_DEPTH, m_depthBuffer, BN_FLOAT);
    *pixelType = ANARI_FLOAT32;
    return m_depthBuffer;
  } else if (channel == "channel.colorGPU") {
    bnFrameBufferRead(m_bnFrameBuffer, BN_FB_COLOR,
                      nullptr, BN_FLOAT4);
    *pixelType = ANARI_FLOAT32_VEC4;
    return bnFrameBufferGetPointer(m_bnFrameBuffer, BN_FB_COLOR);
  } else if (channel == "channel.depthGPU") {
    *pixelType = ANARI_FLOAT32;
    return bnFrameBufferGetPointer(m_bnFrameBuffer, BN_FB_DEPTH);
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
  // no-op (not yet async)
}

bool Frame::ready() const
{
  return true; // not yet async
}

void Frame::wait() const {}

void Frame::cleanup()
{
  delete[] m_colorBuffer;
  delete[] m_depthBuffer;
  m_colorBuffer = nullptr;
  m_depthBuffer = nullptr;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Frame *);
#endif
