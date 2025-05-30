/**
 * Most of this code is derived from the GLM library at https://github.com/g-truc/glm
 *
 * License: https://github.com/g-truc/glm/blob/master/copying.txt
 */

#pragma once

#include <cstdint>

namespace math {
  uint16_t float32ToFloat16(float value);
  float float16ToFloat32(uint16_t value);
}
using math::float32ToFloat16;
using math::float16ToFloat32;

struct float16_t
{
  inline 
  float16_t() = default;

  inline 
  float16_t(uint32_t sign, uint32_t exponent, uint32_t fraction)
    : mBits((sign & 0x01) << 15 | (exponent & 0x1f) << 10 | (fraction & 0x03ff))
  {}

  inline 
  float16_t(float value) : mBits(float32ToFloat16(value)) {}
  // explicit float16_t(float value) : mBits(float32ToFloat16(value)) {}

  template<typename T>
  inline 
  explicit float16_t(T value) : mBits(float32ToFloat16(static_cast<float>(value)))
  {}

  inline 
  operator float() const { return float16ToFloat32(mBits); }
  inline 
  float16_t &operator=(const float f) { mBits = float32ToFloat16(f); return *this; }

  inline 
  static constexpr float16_t fromBits(uint16_t bits) { return float16_t(bits, FromBits); }
  inline 
  uint16_t toBits() const { return mBits; }

  inline 
  bool operator==(const float16_t other) const { return mBits == other.mBits; }
  inline 
  bool operator!=(const float16_t other) const { return mBits != other.mBits; }
  inline 
  bool operator<(const float16_t other) const { return static_cast<float>(*this) < static_cast<float>(other); }
  inline 
  bool operator<=(const float16_t other) const { return static_cast<float>(*this) <= static_cast<float>(other); }
  inline 
  bool operator>(const float16_t other) const { return static_cast<float>(*this) > static_cast<float>(other); }
  inline 
  bool operator>=(const float16_t other) const { return static_cast<float>(*this) >= static_cast<float>(other); }

  inline 
  float16_t operator+() const { return *this; }
  inline 
  float16_t operator-() const { return fromBits(mBits ^ 0x8000); }

  // TODO: Implement math operators in native fp16 precision. For now using fp32.
  inline 
  float16_t operator+(const float16_t other) const { return float16_t(static_cast<float>(*this) + static_cast<float>(other)); }
  inline 
  float16_t operator-(const float16_t other) const { return float16_t(static_cast<float>(*this) - static_cast<float>(other)); }
  inline 
  float16_t operator*(const float16_t other) const { return float16_t(static_cast<float>(*this) * static_cast<float>(other)); }
  inline 
  float16_t operator/(const float16_t other) const { return float16_t(static_cast<float>(*this) / static_cast<float>(other)); }

  inline 
  float16_t operator+=(const float16_t other) { return *this = *this + other; }
  inline 
  float16_t operator-=(const float16_t other) { return *this = *this - other; }
  inline 
  float16_t operator*=(const float16_t other) { return *this = *this * other; }
  inline 
  float16_t operator/=(const float16_t other) { return *this = *this / other; }

  inline 
  constexpr bool isFinite() const noexcept { return exponent() < 31; }
  inline 
  constexpr bool isInf() const noexcept { return exponent() == 31 && mantissa() == 0; }
  inline 
  constexpr bool isNan() const noexcept { return exponent() == 31 && mantissa() != 0; }
  inline 
  constexpr bool isNormalized() const noexcept { return exponent() > 0 && exponent() < 31; }
  inline 
  constexpr bool isDenormalized() const noexcept { return exponent() == 0 && mantissa() != 0; }

private:
  enum Tag
    {
      FromBits
    };

  inline 
  constexpr float16_t(uint16_t bits, Tag) : mBits(bits) {}

  inline 
  constexpr uint16_t mantissa() const noexcept { return mBits & 0x3ff; }
  inline 
  constexpr uint16_t exponent() const noexcept { return (mBits >> 10) & 0x001f; }

  uint16_t mBits;
};


