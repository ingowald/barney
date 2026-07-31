// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/common/barney-common.h"

namespace BARNEY_NS {

  struct Device;

  /*! Cycles-style principled volume coefficients evaluated directly from
      scalar field samples (phase 2 — not baked through a 1D TF). */
  struct PrincipledVolumeParams {
    int     enabled = 0;
    float   density = 1.f;
    vec3f   scatterColor = vec3f(0.8f);
    vec3f   absorptionColor = vec3f(0.f);
    vec3f   emissionColor = vec3f(1.f);
    float   densityThreshold = 0.f;
    float   densityScale = 1.f;
    range1f valueRange = { 0.f, 1.f };
    float   emissionStrength = 0.f;
    float   blackbodyIntensity = 0.f;
    vec3f   blackbodyTint = vec3f(1.f);
    float   temperature = 0.f;

    struct DD {
      int     enabled;
      float   density;
      vec3f   scatterColor;
      vec3f   absorptionColor;
      vec3f   emissionColor;
      float   densityThreshold;
      float   densityScale;
      range1f valueRange;
      float   emissionStrength;
      float   blackbodyIntensity;
      vec3f   blackbodyTint;
      float   temperature;
    };

    inline DD getDD(Device *device) const
    {
      DD dd;
      dd.enabled = enabled;
      dd.density = density;
      dd.scatterColor = scatterColor;
      dd.absorptionColor = absorptionColor;
      dd.emissionColor = emissionColor;
      dd.densityThreshold = densityThreshold;
      dd.densityScale = densityScale;
      dd.valueRange = valueRange;
      dd.emissionStrength = emissionStrength;
      dd.blackbodyIntensity = blackbodyIntensity;
      dd.blackbodyTint = blackbodyTint;
      dd.temperature = temperature;
      (void)device;
      return dd;
    }
  };

  inline __rtc_device
  vec3f principledBlackbodyColor(float kelvin)
  {
    kelvin = clamp(kelvin, 1000.f, 40000.f);
    float t = kelvin / 100.f;
    float r, g, b;
    if (t <= 66.f) {
      r = 1.f;
      g = clamp(0.390081578f * logf(t) - 0.631841444f, 0.f, 1.f);
    } else {
      r = clamp(1.292936186f * powf(t - 60.f, -0.1332047592f), 0.f, 1.f);
      g = clamp(1.129890860f * powf(t - 60.f, -0.0755148492f), 0.f, 1.f);
    }
    if (t >= 66.f)
      b = 1.f;
    else if (t <= 19.f)
      b = 0.f;
    else
      b = clamp(0.543206789f * logf(t - 10.f) - 1.196254089f, 0.f, 1.f);
    return vec3f(r, g, b);
  }

  inline __rtc_device
  float principledDensityFactor(float scalar,
                                const PrincipledVolumeParams::DD &p)
  {
    float s = clamp(scalar, p.valueRange.lower, p.valueRange.upper);
    if (p.densityThreshold > 0.f && s < p.densityThreshold)
      return 0.f;
    return p.density * max(s, 0.f) * p.densityScale;
  }

  inline __rtc_device
  vec3f principledSigmaS(float scalar,
                          const PrincipledVolumeParams::DD &p)
  {
    return principledDensityFactor(scalar, p) * p.scatterColor;
  }

  inline __rtc_device
  vec3f principledSigmaA(float scalar,
                          const PrincipledVolumeParams::DD &p)
  {
    return principledDensityFactor(scalar, p) * p.absorptionColor;
  }

  inline __rtc_device
  float principledSigmaT(float scalar,
                           const PrincipledVolumeParams::DD &p)
  {
    vec3f sigma_t = principledSigmaS(scalar, p) + principledSigmaA(scalar, p);
    return reduce_max(sigma_t);
  }

  inline __rtc_device
  vec4f principledSampleMap(float scalar,
                            const PrincipledVolumeParams::DD &p)
  {
    vec3f sigma_s = principledSigmaS(scalar, p);
    vec3f sigma_a = principledSigmaA(scalar, p);
    vec3f sigma_t = sigma_s + sigma_a;
    float sigma_t_scalar = reduce_max(sigma_t);
    vec3f tint = vec3f(1.f);
    if (sigma_t_scalar > 0.f) {
      float ms = reduce_max(sigma_s);
      tint = ms > 0.f ? sigma_s / ms : vec3f(1.f);
    }
    return vec4f(tint, sigma_t_scalar);
  }

  inline __rtc_device
  vec3f principledEmission(float scalar,
                           const PrincipledVolumeParams::DD &p)
  {
    float d = principledDensityFactor(scalar, p);
    vec3f Le = p.emissionStrength * p.emissionColor * d;
    if (p.blackbodyIntensity > 0.f && p.temperature > 0.f)
      Le += p.blackbodyIntensity * p.blackbodyTint
        * principledBlackbodyColor(p.temperature) * d;
    return Le;
  }

  inline __rtc_device
  float principledMajorant(range1f r,
                           const PrincipledVolumeParams::DD &p)
  {
    if (r.lower > r.upper)
      return 0.f;
    float m = 0.f;
    const int steps = 8;
    for (int i = 0; i <= steps; ++i) {
      float t = float(i) / float(steps);
      float s = r.lower + t * r.span();
      m = max(m, principledSigmaT(s, p));
    }
    return m;
  }

}
