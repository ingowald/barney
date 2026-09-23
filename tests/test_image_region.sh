#!/usr/bin/env bash
# Regression test: imageRegion remaps screen coordinates without clamping.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

cat > "$TMPDIR_TEST/main.cpp" <<'EOF'
#include <cstdio>

#define __rtc_both
#include "native/ImageRegion.h"

static int failures = 0;

static void expect(float actual, float expected, const char *message)
{
  if (actual != expected) {
    std::printf("FAIL: %s: got %.9g, expected %.9g\n",
                message, actual, expected);
    ++failures;
  }
}

int main()
{
  const float width = 8.f;
  const float height = 4.f;
  const float lowerX = 1.f / width;
  const float lowerY = 1.f / height;
  const float upperX = 1.f + 1.f / width;
  const float upperY = 1.f + 1.f / height;

  float screenX = BARNEY_NS::native::remapImageRegionCoordinate(
      lowerX, upperX, 0.5f / width);
  float screenY = BARNEY_NS::native::remapImageRegionCoordinate(
      lowerY, upperY, 0.5f / height);

  expect(screenX, 1.5f / width,
         "one-pixel horizontal translation shifts the sample one pixel");
  expect(screenY, 1.5f / height,
         "one-pixel vertical translation shifts the sample one pixel");

  screenX = BARNEY_NS::native::remapImageRegionCoordinate(
      lowerX, upperX, 1.f);
  screenY = BARNEY_NS::native::remapImageRegionCoordinate(
      lowerY, upperY, 1.f);

  expect(screenX, 1.f + 1.f / width,
         "horizontal remap extrapolates beyond the unit interval");
  expect(screenY, 1.f + 1.f / height,
         "vertical remap extrapolates beyond the unit interval");

  return failures == 0 ? 0 : 1;
}
EOF

g++ -std=c++17 -Wall -Wextra -Werror \
  -DBARNEY_NS=barney_test \
  -I"$REPO_ROOT/barney" \
  "$TMPDIR_TEST/main.cpp" -o "$TMPDIR_TEST/test_image_region"

"$TMPDIR_TEST/test_image_region"
