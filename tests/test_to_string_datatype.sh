#!/usr/bin/env bash
# Regression test: to_string(BNDataType) must handle all public enum values.
#
# Bug: to_string(BNDataType) in barney/common/Data.cpp threw
#   "#bn internal error: to_string not implemented for numerical BNDataType #N"
# for the valid public values BN_INT16 family, BN_UINT16 family, BN_UFIXED8,
# BN_UFIXED8_RGBA_SRGB and BN_UFIXED16, producing a misleading "internal
# error" in place of a useful message naming the type.
#
# The full project cannot build standalone (CUDA/OptiX/Embree + submodules),
# so this harness extracts the *actual* to_string(BNDataType) implementation
# from barney/common/Data.cpp, compiles it against the real BNDataType enum
# from barney/include/barney.h, and exercises every fixed value.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_CPP="$REPO_ROOT/barney/common/Data.cpp"

TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

# Extract the real to_string(BNDataType) function from Data.cpp.
sed -n '/std::string to_string(BNDataType type)/,/^}$/p' "$DATA_CPP" \
  > "$TMPDIR_TEST/to_string.inc"

if ! grep -q 'switch' "$TMPDIR_TEST/to_string.inc"; then
  echo "FAIL: could not extract to_string(BNDataType) from $DATA_CPP" >&2
  exit 1
fi

# barney.h is a CMake template: materialize #cmakedefine01 lines as 0.
sed 's/^#cmakedefine01 \([A-Za-z_0-9]*\)/#define \1 0/' \
  "$REPO_ROOT/barney/include/barney.h" > "$TMPDIR_TEST/barney.h"

cat > "$TMPDIR_TEST/main.cpp" <<'EOF'
#include <cstdio>
#include <stdexcept>
#include <string>

#include "barney.h"

// The implementation under test, extracted verbatim from
// barney/common/Data.cpp:
#include "to_string.inc"

using namespace BARNEY_NS;

static int failures = 0;

static void expect(BNDataType type, const char *expected)
{
  try {
    std::string got = to_string(type);
    if (got != expected) {
      printf("FAIL: to_string(%d) == \"%s\", expected \"%s\"\n",
             int(type), got.c_str(), expected);
      ++failures;
    } else {
      printf("ok:   to_string(%d) == \"%s\"\n", int(type), got.c_str());
    }
  } catch (const std::exception &e) {
    printf("FAIL: to_string(%d) threw: %s\n", int(type), e.what());
    ++failures;
  }
}

int main()
{
  // Values that previously threw "to_string not implemented":
  expect(BN_INT16,              "BN_INT16");
  expect(BN_INT16_VEC2,         "BN_INT16_VEC2");
  expect(BN_INT16_VEC3,         "BN_INT16_VEC3");
  expect(BN_INT16_VEC4,         "BN_INT16_VEC4");
  expect(BN_UINT16,             "BN_UINT16");
  expect(BN_UINT16_VEC2,        "BN_UINT16_VEC2");
  expect(BN_UINT16_VEC3,        "BN_UINT16_VEC3");
  expect(BN_UINT16_VEC4,        "BN_UINT16_VEC4");
  expect(BN_UFIXED8,            "BN_UFIXED8");
  expect(BN_UFIXED8_RGBA_SRGB,  "BN_UFIXED8_RGBA_SRGB");
  expect(BN_UFIXED16,           "BN_UFIXED16");

  // Regression checks for cases that already worked:
  expect(BN_DATA_UNDEFINED,     "BN_DATA_UNDEFINED");
  expect(BN_INT8,               "BN_INT8");
  expect(BN_UINT8_VEC4,         "BN_UINT8_VEC4");
  expect(BN_FLOAT32,            "BN_FLOAT32");
  expect(BN_FLOAT64_VEC4,       "BN_FLOAT64_VEC4");
  expect(BN_UFIXED8_RGBA,       "BN_UFIXED8_RGBA");

  if (failures) {
    printf("%d case(s) FAILED\n", failures);
    return 1;
  }
  printf("all cases passed\n");
  return 0;
}
EOF

g++ -std=c++17 -Wall \
  -I"$TMPDIR_TEST" \
  "$TMPDIR_TEST/main.cpp" -o "$TMPDIR_TEST/test_to_string"

"$TMPDIR_TEST/test_to_string"
