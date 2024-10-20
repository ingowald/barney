// Copyright 2023 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// std
#include <algorithm>
#include <chrono>
#include <iostream>
// cuda
#include <cuda_runtime_api.h>
// thrust
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

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
  return m_valid;
}

BarneyGlobalState *Frame::deviceState() const
{
  return (BarneyGlobalState *)helium::BaseObject::m_state;
}

void Frame::commit()
{
  cleanup();

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
  cudaMallocManaged((void **)&m_colorBuffer, numPixels * perPixelBytes);
  cudaMallocManaged((void **)&m_bnPixelBuffer, numPixels * sizeof(uint32_t));
  if (m_depthType == ANARI_FLOAT32)
    cudaMallocManaged((void **)&m_depthBuffer, numPixels * sizeof(float));

  bnFrameBufferResize(
      m_bnFrameBuffer, size.x, size.y, m_bnPixelBuffer, m_depthBuffer);
  bnSet1i(m_bnFrameBuffer,
      "showCrosshairs",
      m_renderer ? int(m_renderer->crosshairs()) : false);
  bnCommit(m_bnFrameBuffer);
  m_frameData.size = size;
  m_frameData.totalPixels = numPixels;
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

  if (state->commitBufferLastFlush() > m_frameLastRendered) {
    bnAccumReset(m_bnFrameBuffer);
  }

  auto model = m_world->makeCurrent();

  m_frameLastRendered = helium::newTimeStamp();
  state->currentFrame = this;

  const int pixelSamples = std::max(m_renderer->pixelSamples(), 1);
  const float radiance = m_renderer->radiance();
  bnSetRadiance(model, 0, radiance
#if 0
                /* iw- no idea where this factor came from, but it
                   probably shouldn't be here */
                / 10.f
#endif
                );

  bnRender(model, m_camera->barneyCamera(), m_bnFrameBuffer, pixelSamples);

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
    return m_colorBuffer;
  } else if (channel == "channel.depth" && m_depthBuffer) {
    *pixelType = ANARI_FLOAT32;
    return m_depthBuffer;
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
    cudaMemcpy(
        m_colorBuffer,
        m_bnPixelBuffer,
        m_frameData.totalPixels * sizeof(uint32_t),
        cudaMemcpyDefault);
  } else if (m_colorType == ANARI_UFIXED8_RGBA_SRGB) {
    auto numPixels = m_frameData.totalPixels;
    auto *src = (math::byte4 *)m_bnPixelBuffer;
    auto *dst = (math::byte4 *)m_colorBuffer;
    thrust::transform(thrust::device,
        src,
        src + numPixels,
        dst,
        [] __device__(math::byte4 p) {
          auto f =
              math::float4(p.x / 255.f, p.y / 255.f, p.z / 255.f, p.w / 255.f);
          f.x = std::pow(f.x, 1.f / 2.2f);
          f.y = std::pow(f.y, 1.f / 2.2f);
          f.z = std::pow(f.z, 1.f / 2.2f);
          return math::byte4(f.x * 255, f.y * 255, f.z * 255, f.w * 255);
        });
  } else if (m_colorType == ANARI_FLOAT32_VEC4) {
    auto numPixels = m_frameData.totalPixels;
#if 1
    const uint32_t *src = (const uint32_t *)m_bnPixelBuffer;
    math::float4 *dst  = (math::float4   *)m_colorBuffer;
    for (int i=0;i<numPixels;i++) {
      uint32_t src_i = src[i];
      const uint8_t *bytes = (const uint8_t *)&src_i;
      dst[i].x = bytes[0]/255.f;
      dst[i].y = bytes[1]/255.f;
      dst[i].z = bytes[2]/255.f;
      dst[i].w = bytes[3]/255.f;
    }
#else
    auto *src = (math::byte4 *)m_bnPixelBuffer;
    auto *dst = (math::float4 *)m_colorBuffer;
    thrust::transform(thrust::device,
        src,
        src + numPixels,
        dst,
        [] __device__(math::byte4 p) {
          return math::float4(
              p.x / 255.f, p.y / 255.f, p.z / 255.f, p.w / 255.f);
        });
    printf("src %i %i %i %i\n",src->x,src->y,src->z,src->w);
#endif
  }
}

void Frame::cleanup()
{
  cudaFree(m_bnPixelBuffer);
  cudaFree(m_colorBuffer);
  cudaFree(m_depthBuffer);
  m_bnPixelBuffer = nullptr;
  m_colorBuffer = nullptr;
  m_depthBuffer = nullptr;
}

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Frame *);
