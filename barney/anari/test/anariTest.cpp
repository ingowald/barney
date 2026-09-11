// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <array>
#include <cstring>
#include <vector>
// anari
#define ANARI_EXTENSION_UTILITY_IMPL
#include "anari/anari_cpp.hpp"
#include "anari/anari_cpp/ext/std.h"
// stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using uvec2 = std::array<unsigned int, 2>;
using uvec3 = std::array<unsigned int, 3>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using box3 = std::array<vec3, 2>;

void statusFunc(const void *userData,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  (void)userData;
  (void)device;
  (void)source;
  (void)sourceType;
  (void)code;
  if (severity == ANARI_SEVERITY_FATAL_ERROR) {
    fprintf(stderr, "[FATAL] %s\n", message);
  } else if (severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "[ERROR] %s\n", message);
  } else if (severity == ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[WARN ] %s\n", message);
  } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
    fprintf(stderr, "[PERF ] %s\n", message);
  } else if (severity == ANARI_SEVERITY_INFO) {
    fprintf(stderr, "[INFO ] %s\n", message);
  } else if (severity == ANARI_SEVERITY_DEBUG) {
    fprintf(stderr, "[DEBUG] %s\n", message);
  }
}

template <typename T>
static T getPixelValue(uvec2 coord, int width, const T *buf)
{
  return buf[coord[1] * width + coord[0]];
}

int main(int argc, const char **argv)
{
  (void)argc;
  (void)argv;
  stbi_flip_vertically_on_write(1);

  // image size
  uvec2 imgSize = {1024 /*width*/, 768 /*height*/};

  // camera
  vec3 cam_pos = {0.f, 0.f, -2.f};
  vec3 cam_up = {0.f, 1.f, 0.f};
  vec3 cam_view = {0.1f, 0.f, 1.f};

  // triangle mesh array
  vec3 vertex[] = {{-1.0f, -1.0f, 3.0f},
      {-1.0f, 1.0f, 3.0f},
      {1.0f, -1.0f, 3.0f},
      {0.1f, 0.1f, 0.3f}};
  vec4 color[] = {{0.9f, 0.5f, 0.5f, 1.0f},
      {0.8f, 0.8f, 0.8f, 1.0f},
      {0.8f, 0.8f, 0.8f, 1.0f},
      {0.5f, 0.9f, 0.5f, 1.0f}};
  uvec3 index[] = {{0, 1, 2}, {1, 2, 3}};

  printf("initialize ANARI...");
#if 1
  anari::Library lib = anari::loadLibrary("barney", statusFunc);
#else
  anari::Library lib = anari::loadLibrary("helide", statusFunc);
#endif

  anari::Extensions extensions =
      anari::extension::getDeviceExtensionStruct(lib, "default");

  if (!extensions.ANARI_KHR_GEOMETRY_TRIANGLE)
    printf("WARNING: device doesn't support ANARI_KHR_GEOMETRY_TRIANGLE\n");
  if (!extensions.ANARI_KHR_CAMERA_PERSPECTIVE)
    printf("WARNING: device doesn't support ANARI_KHR_CAMERA_PERSPECTIVE\n");
  if (!extensions.ANARI_KHR_MATERIAL_MATTE)
    printf("WARNING: device doesn't support ANARI_KHR_MATERIAL_MATTE\n");

  anari::Device d = anari::newDevice(lib, "default");

  printf("done!\n");
  printf("setting up camera...");

  // create and setup camera
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  anari::setParameter(
      d, camera, "aspect", (float)imgSize[0] / (float)imgSize[1]);
  anari::setParameter(d, camera, "position", cam_pos);
  anari::setParameter(d, camera, "direction", cam_view);
  anari::setParameter(d, camera, "up", cam_up);
  anari::commitParameters(
      d, camera); // commit objects to indicate setting parameters is done

  printf("done!\n");
  printf("setting up scene...");

  // The world to be populated with renderable objects
  auto world = anari::newObject<anari::World>(d);

  // create and setup surface and mesh
  auto mesh = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(d, mesh, "vertex.position", vertex, 4);
  anari::setParameterArray1D(d, mesh, "vertex.color", color, 4);
  anari::setParameterArray1D(d, mesh, "primitive.index", index, 2);
  anari::commitParameters(d, mesh);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", "color");
  anari::commitParameters(d, mat);

  // put the mesh into a surface
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", mesh);
  anari::setAndReleaseParameter(d, surface, "material", mat);
  anari::setParameter(d, surface, "id", 2u);
  anari::commitParameters(d, surface);

  std::vector<ANARISurface> surfaces;
  surfaces.push_back(surface);
  anari::release(d, surface);

  // ---------------------------------------------------------------------
  // type-matrix meshes: one quad per data route (see anari/TexelPolicy.h)
  // ---------------------------------------------------------------------
  auto quadGeometry = [&](float ox, float oy) {
    vec3 v[] = {{ox - 1.f, oy - 1.f, 3.f},
                {ox - 1.f, oy + 1.f, 3.f},
                {ox + 1.f, oy - 1.f, 3.f},
                {ox + 1.f, oy + 1.f, 3.f}};
    uvec3 idx[] = {{0u, 1u, 2u}, {1u, 2u, 3u}};
    auto m = anari::newObject<anari::Geometry>(d, "triangle");
    anari::setParameterArray1D(d, m, "vertex.position", v, 4);
    anari::setParameterArray1D(d, m, "primitive.index", idx, 2);
    anari::commitParameters(d, m);
    return m;
  };

  auto addSurface = [&](ANARIGeometry geom, ANARIMaterial matl) {
    auto s = anari::newObject<anari::Surface>(d);
    anari::setAndReleaseParameter(d, s, "geometry", geom);
    anari::setAndReleaseParameter(d, s, "material", matl);
    anari::commitParameters(d, s);
    surfaces.push_back(s);
    anari::release(d, s);
  };

  auto matteFromColorAttribute = [&]() {
    auto m = anari::newObject<anari::Material>(d, "matte");
    anari::setParameter(d, m, "color", "color");
    anari::commitParameters(d, m);
    return m;
  };

  { // pristine ufixed8 colors (read-time typedRead expansion)
    ANARIArray1D colors = anariNewArray1D(
        d, nullptr, nullptr, nullptr, ANARI_UFIXED8_VEC4, 4);
    const uint8_t rgba8[] = {255, 80, 80, 255,
                             80, 255, 80, 255,
                             80, 80, 255, 255,
                             255, 255, 80, 255};
    memcpy(anariMapArray(d, colors), rgba8, sizeof(rgba8));
    anariUnmapArray(d, colors);
    auto m = quadGeometry(-2.5f, 0.f);
    anari::setParameter(d, m, "vertex.color", colors);
    anari::commitParameters(d, m);
    anari::release(d, colors);
    addSurface(m, matteFromColorAttribute());
  }

  { // primitive sampler over a pristine float32 array (scalar reads
    // as (x,0,0,1) per ANARI semantics - solid red quad expected)
    float primData[] = {0.7f, 0.25f};
    auto arr = anari::newArray1D(d, primData, 2);
    auto sampler = anari::newObject<anari::Sampler>(d, "primitive");
    anari::setAndReleaseParameter(d, sampler, "array", arr);
    anari::setParameter(d, sampler, "inAttribute", "color");
    anari::commitParameters(d, sampler);
    auto m = quadGeometry(-2.5f, 1.6f);
    auto matl = anari::newObject<anari::Material>(d, "matte");
    anari::setAndReleaseParameter(d, matl, "color", sampler);
    anari::commitParameters(d, matl);
    addSurface(m, matl);
  }

  { // pristine float3 colors
    vec3 color[] = {{0.9f, 0.5f, 0.5f},
                    {0.5f, 0.9f, 0.5f},
                    {0.5f, 0.5f, 0.9f},
                    {0.9f, 0.9f, 0.5f}};
    auto m = quadGeometry(2.5f, 0.f);
    anari::setParameterArray1D(d, m, "vertex.color", color, 4);
    anari::commitParameters(d, m);
    addSurface(m, matteFromColorAttribute());
  }

  { // float64 colors - upload-time demotion to float32
    double color[][3] = {{0.9, 0.5, 0.5},
                        {0.5, 0.9, 0.5},
                        {0.5, 0.5, 0.9},
                        {0.9, 0.9, 0.5}};
    auto m = quadGeometry(2.5f, -1.6f);
    anari::setParameterArray1D(d, m, "vertex.color", color, 4);
    anari::commitParameters(d, m);
    addSurface(m, matteFromColorAttribute());
  }

  { // 2-component float texture: no backend implements a 2-channel
    // float texel format, so this must route through the float4
    // demote rather than reaching rtcore (regression: it used to
    // route pristine and abort the host in optixDenoiser-style
    // uninitialized-descriptor UB)
    ANARIArray2D img = anariNewArray2D(
        d, nullptr, nullptr, nullptr, ANARI_FLOAT32_VEC2, 2u, 2u);
    const float texels[] = {1.f, 0.f,  0.f, 1.f,
                            0.5f, 0.5f,  0.25f, 0.75f};
    memcpy(anariMapArray(d, img), texels, sizeof(texels));
    anariUnmapArray(d, img);
    auto sampler = anari::newObject<anari::Sampler>(d, "image2D");
    anari::setAndReleaseParameter(d, sampler, "image", img);
    anari::setParameter(d, sampler, "inAttribute", "attribute0");
    anari::commitParameters(d, sampler);

    auto m = quadGeometry(2.5f, 1.6f);
    std::array<float, 2> uv[] = {{0.f, 0.f}, {1.f, 0.f},
                                {1.f, 1.f}, {0.f, 1.f}};
    anari::setParameterArray1D(d, m, "vertex.attribute0", uv, 4);
    anari::commitParameters(d, m);

    auto matl = anari::newObject<anari::Material>(d, "matte");
    anari::setAndReleaseParameter(d, matl, "color", sampler);
    anari::commitParameters(d, matl);
    addSurface(m, matl);
  }

  { // pristine float16 texture - hardware promotes half on fetch
    ANARIArray2D img = anariNewArray2D(
        d, nullptr, nullptr, nullptr, ANARI_FLOAT16_VEC4, 2u, 2u);
    const uint16_t h_one = 0x3C00, h_zero = 0x0000;
    const uint16_t texels[] = {h_one, h_zero, h_zero, h_one,
                              h_zero, h_one, h_zero, h_one,
                              h_zero, h_zero, h_one, h_one,
                              h_one, h_one, h_one, h_one};
    memcpy(anariMapArray(d, img), texels, sizeof(texels));
    anariUnmapArray(d, img);
    auto sampler = anari::newObject<anari::Sampler>(d, "image2D");
    anari::setAndReleaseParameter(d, sampler, "image", img);
    anari::setParameter(d, sampler, "inAttribute", "attribute0");
    anari::commitParameters(d, sampler);

    auto m = quadGeometry(-2.5f, -1.6f);
    std::array<float, 2> uv[] = {{0.f, 0.f}, {1.f, 0.f},
                                {1.f, 1.f}, {0.f, 1.f}};
    anari::setParameterArray1D(d, m, "vertex.attribute0", uv, 4);
    anari::commitParameters(d, m);

    auto matl = anari::newObject<anari::Material>(d, "matte");
    anari::setAndReleaseParameter(d, matl, "color", sampler);
    anari::commitParameters(d, matl);
    addSurface(m, matl);
  }

  { // float16 colors - pristine (was upload-time demotion before
    // native half; must render identically - regression check)
    ANARIArray1D ah = anariNewArray1D(
        d, nullptr, nullptr, nullptr, ANARI_FLOAT16_VEC4, 4);
    uint16_t *mapped = (uint16_t *)anariMapArray(d, ah);
    const uint16_t half_1_0 = 0x3C00, half_0_5 = 0x3800,
                   half_0_25 = 0x3400, half_0_75 = 0x3B00;
    const uint16_t halfs[] = {half_1_0, half_0_25, half_0_25, half_1_0,
                              half_0_25, half_1_0, half_0_25, half_1_0,
                              half_0_25, half_0_25, half_1_0, half_1_0,
                              half_0_75, half_0_5, half_0_5, half_0_75};
    memcpy(mapped, halfs, sizeof(halfs));
    anariUnmapArray(d, ah);
    auto m = quadGeometry(0.f, -1.6f);
    anari::setParameter(d, m, "vertex.color", ah);
    anari::commitParameters(d, m);
    anari::release(d, ah);
    addSurface(m, matteFromColorAttribute());
  }

  { // sRGB RA vertex colors: ANARI toFloat4 is (srgb(r),0,0,a), so
    // this must demote to float4, not float2. (255,255) is opaque
    // red; yellow would mean alpha leaked into G.
    ANARIArray1D colors = anariNewArray1D(
        d, nullptr, nullptr, nullptr, ANARI_UFIXED8_RA_SRGB, 4);
    const uint8_t ra[] = {255, 255, 255, 255, 255, 255, 255, 255};
    memcpy(anariMapArray(d, colors), ra, sizeof(ra));
    anariUnmapArray(d, colors);
    auto m = quadGeometry(0.f, 3.2f);
    anari::setParameter(d, m, "vertex.color", colors);
    anari::commitParameters(d, m);
    anari::release(d, colors);
    addSurface(m, matteFromColorAttribute());
  }

  { // sRGB image2D sampler - hardware decode at sampling time
    uint8_t texels[] = {255, 0, 0, 255,
                        0, 255, 0, 255,
                        0, 0, 255, 255,
                        255, 255, 255, 255};
    ANARIArray2D img = anariNewArray2D(
        d, nullptr, nullptr, nullptr, ANARI_UFIXED8_VEC4, 2u, 2u);
    memcpy(anariMapArray(d, img), texels, sizeof(texels));
    anariUnmapArray(d, img);
    auto sampler = anari::newObject<anari::Sampler>(d, "image2D");
    anari::setAndReleaseParameter(d, sampler, "image", img);
    anari::setParameter(d, sampler, "inAttribute", "attribute0");
    anari::commitParameters(d, sampler);

    auto m = quadGeometry(0.f, 1.6f);
    std::array<float, 2> uv[] = {{0.f, 0.f}, {1.f, 0.f},
                                {1.f, 1.f}, {0.f, 1.f}};
    anari::setParameterArray1D(d, m, "vertex.attribute0", uv, 4);
    anari::commitParameters(d, m);

    auto matl = anari::newObject<anari::Material>(d, "matte");
    anari::setAndReleaseParameter(d, matl, "color", sampler);
    anari::commitParameters(d, matl);
    addSurface(m, matl);
  }

  // put the surfaces directly onto the world
  anari::setParameterArray1D(d, world, "surface", surfaces.data(),
                             (uint64_t)surfaces.size());
  anari::setParameter(d, world, "id", 3u);

  anari::commitParameters(d, world);

  printf("done!\n");

  // print out world bounds
  box3 worldBounds;
  if (anari::getProperty(d, world, "bounds", worldBounds, ANARI_WAIT)) {
    printf("\nworld bounds: ({%f, %f, %f}, {%f, %f, %f}\n\n",
        worldBounds[0][0],
        worldBounds[0][1],
        worldBounds[0][2],
        worldBounds[1][0],
        worldBounds[1][1],
        worldBounds[1][2]);
  } else {
    printf("\nworld bounds not returned\n\n");
  }

  printf("setting up renderer...");

  // create renderer
  auto renderer = anari::newObject<anari::Renderer>(d, "default");
  // objects can be named for easier identification in debug output etc.
  anari::setParameter(d, renderer, "name", "MainRenderer");
  anari::setParameter(d, renderer, "ambientRadiance", 1.f);
  anari::commitParameters(d, renderer);

  printf("done!\n");

  // create and setup frame
  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", imgSize);
  anari::setParameter(d, frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(d, frame, "channel.primitiveId", ANARI_UINT32);
  anari::setParameter(d, frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(d, frame, "channel.instanceId", ANARI_UINT32);

  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "world", world);

  anari::commitParameters(d, frame);

  printf("rendering initial frame to firstFrame.png...");

  // render one frame
  anari::render(d, frame);
  anari::wait(d, frame);

  // access frame and write its content as PNG file
  auto fb = anari::map<uint32_t>(d, frame, "channel.color");
  stbi_write_png("anariTest.png",
      int(fb.width),
      int(fb.height),
      4,
      fb.data,
      4 * int(fb.width));
  anari::unmap(d, frame, "channel.color");

  printf("done!\n");

  printf("\ncleaning up objects...");

  // final cleanups
  anari::release(d, frame);
  anari::release(d, d);
  anari::unloadLibrary(lib);

  printf("done!\n");

  return 0;
}
