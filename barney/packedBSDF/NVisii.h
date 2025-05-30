// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "barney/render/DG.h"
#include "barney/render/floatN.h"

#define MIN_ALPHA .002f
#define SMALL_EPSILON 2e-10f
#define IMPORTANCE_SAMPLE_BRDF 1

namespace BARNEY_NS {
  namespace render {

      inline __rtc_device float pow2(float f) { return f*f; }
      inline __rtc_device float pow5(float f) { return pow2(pow2(f))*f; }
      inline __rtc_device float mix(float a, float b, float f) { return (1.f-f)*a + f*b; }
      inline __rtc_device vec3f mix(vec3f a, vec3f b, vec3f f)
      { return vec3f(mix(a.x,b.x,f.x),mix(a.y,b.y,f.y),mix(a.z,b.z,f.z)); }
      inline __rtc_device float heaviside(float f) { return (f<0.f)?0.f:1.f; }

    
    namespace packedBSDF {
      namespace nvisii {
        inline __rtc_device float lcg_randomf(Random &r) { return r(); }

#define DISNEY_DIFFUSE_BRDF 0
#define DISNEY_GLOSSY_BRDF 1
#define DISNEY_CLEARCOAT_BRDF 2
#define DISNEY_TRANSMISSION_BRDF 3

        struct DisneyMaterial {
          vec3f base_color;
          vec3f subsurface_color;
          float metallic;

          float specular;
          float roughness;
          float specular_tint;
          float anisotropy;

          float sheen;
          float sheen_tint;
          float clearcoat;
          float clearcoat_gloss;

          float ior;
          float specular_transmission;
          float transmission_roughness;
          float flatness;
          float alpha;
        };

        inline
        __rtc_device bool same_hemisphere(const vec3f &w_o, const vec3f &w_i, const vec3f &n) {
          return dot(w_o, n) * dot(w_i, n) > 0.f;
        } 

        inline
        __rtc_device bool relative_ior(const vec3f &w_o, const vec3f &n, float ior, float &eta_o, float &eta_i)
        {
          bool entering = dot(w_o, n) > 0.f;
          eta_i = entering ? 1.f : ior;
          eta_o = entering ? ior : 1.f;
          return entering;
        }

        // Sample the hemisphere using a cosine weighted distribution,
        // returns a vector in a hemisphere oriented about (0, 0, 1)
        inline
        __rtc_device vec3f cos_sample_hemisphere(vec2f u) {
          vec2f s = 2.f * u - vec2f(1.f);
          vec2f d;
          float radius = 0.f;
          float theta = 0.f;
          if (s.x == 0.f && s.y == 0.f) {
            d = s;
          } else {
            if (fabs(s.x) > fabs(s.y)) {
              radius = s.x;
              theta  = (float)M_PI / 4.f * (s.y / s.x);
            } else {
              radius = s.y;
              theta  = (float)M_PI / 2.f - (float)M_PI / 4.f * (s.x / s.y);
            }
          }
          d = radius * vec2f(cosf(theta), sinf(theta));
          return vec3f(d.x, d.y, sqrtf(max(0.f, 1.f - d.x * d.x - d.y * d.y)));
        }

        inline
        __rtc_device vec3f spherical_dir(float sin_theta, float cos_theta, float phi) {
          return vec3f{ sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta };
        }

        inline
        __rtc_device float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
          float f = n_f * pdf_f;
          float g = n_g * pdf_g;
          return (f * f) / (f * f + g * g);
        }

        inline
        __rtc_device float schlick_weight(float cos_theta) {
          return powf(saturate(1.f - cos_theta), 5.f);
        }

        // Complete Fresnel Dielectric computation, for transmission at ior near 1
        // they mention having issues with the Schlick approximation.
        // eta_i: material on incident side's ior
        // eta_t: material on transmitted side's ior
        inline
        __rtc_device float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
          float g = pow2(eta_t) / pow2(eta_i) - 1.f + pow2(cos_theta_i);
          if (g < 0.f) {
            return 1.f;
          }
          return 0.5f * pow2(g - cos_theta_i) / pow2(g + cos_theta_i)
            * (1.f + pow2(cos_theta_i * (g + cos_theta_i) - 1.f) / pow2(cos_theta_i * (g - cos_theta_i) + 1.f));
        }

        // D_GTR1: Generalized Trowbridge-Reitz with gamma=1
        // Burley notes eq. 4
        inline
        __rtc_device float gtr_1(float cos_theta_h, float alpha) {
          if (alpha >= 1.f) {
            return (float)M_1_PI;
          }
          float alpha_sqr = alpha * alpha;
          return (float)M_1_PI * (alpha_sqr - 1.f) / (logf(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
        }

        // D_GTR2: Generalized Trowbridge-Reitz with gamma=2
        // Burley notes eq. 8
        inline
        __rtc_device float gtr_2(float cos_theta_h, float alpha, bool dbg=0)
        {
          float alpha_sqr = alpha * alpha;
          float den1 = 1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h;
          float den2 = max(den1, SMALL_EPSILON);
          // float ret = (float)M_1_PI * 1.f / den2;
          float ret = (float)M_1_PI * alpha_sqr / den2;
          if (dbg)
            printf("gtr_2 alpha %f cos_theta_h %f den1 %f den2 %f ret %f\n",
                   alpha,cos_theta_h,den1,den2,ret);
          return ret;
        }

        // D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
        // Burley notes eq. 13
        inline
        __rtc_device float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, vec2f alpha) {
          return (float)M_1_PI / max((alpha.x * alpha.y * pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n)), SMALL_EPSILON);
        }

        inline
        __rtc_device float smith_shadowing_ggx(float n_dot_o, float alpha_g) {
          float a = alpha_g * alpha_g;
          float b = n_dot_o * n_dot_o;
          return 1.f / (n_dot_o + sqrtf(a + b - a * b));
        }

        inline
        __rtc_device float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, vec2f alpha) {
          return 1.f / (n_dot_o + sqrtf(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
        }

        // Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
        inline
        __rtc_device vec3f sample_lambertian_dir(const vec3f &n, const vec3f &v_x, const vec3f &v_y, const vec2f &s) {
          const vec3f hemi_dir = normalize(cos_sample_hemisphere(s));
          return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
        }

        // Sample the microfacet normal vectors for the various microfacet distributions
        inline
        __rtc_device vec3f sample_gtr_1_h(const vec3f &n, const vec3f &v_x, const vec3f &v_y, float alpha, const vec2f &s) {
          float phi_h = 2.f * (float)M_PI * s.x;
          float alpha_sqr = alpha * alpha;
          float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
          float cos_theta_h = sqrtf(cos_theta_h_sqr);
          float sin_theta_h = 1.f - cos_theta_h_sqr;
          vec3f hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
          return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
        }

        inline
        __rtc_device vec3f sample_gtr_2_h(const vec3f &n, const vec3f &v_x, const vec3f &v_y, float alpha, const vec2f &s) {
          float phi_h = 2.f * (float)M_PI * s.x;
          float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
          float cos_theta_h = sqrtf(cos_theta_h_sqr);
          float sin_theta_h = 1.f - cos_theta_h_sqr;
          vec3f hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
          return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
        }

        inline
        __rtc_device vec3f sample_gtr_2_aniso_h(const vec3f &n, const vec3f &v_x, const vec3f &v_y, const vec2f &alpha, const vec2f &s) {
          float x = 2.f * (float)M_PI * s.x;
          vec3f w_h = sqrtf(s.y / (1.f - s.y)) * (alpha.x * cosf(x) * v_x + alpha.y * sinf(x) * v_y) + n;
          return normalize(w_h);
        }

        inline
        __rtc_device float lambertian_pdf(const vec3f &w_i, const vec3f &n) {
          float d = dot(w_i, n);
          if (d > 0.f) {
            return d * (float)M_1_PI;
          }
          return 0.f;
        }

        inline
        __rtc_device float gtr_1_pdf(const vec3f &w_o, const vec3f &w_i, const vec3f &w_h, const vec3f &n, float alpha) {
          if (!same_hemisphere(w_o, w_i, n)) {
            return 0.f;
          }
          float cos_theta_h = dot(n, w_h);
          float d = gtr_1(cos_theta_h, alpha);
          return d * cos_theta_h / (4.f * dot(w_o, w_h));
        }

        inline
        __rtc_device float gtr_2_pdf(const vec3f &w_o, const vec3f &w_i, const vec3f &w_h, const vec3f &n, float alpha) {
          if (!same_hemisphere(w_o, w_i, n)) {
            return 0.f;
          }
          float cos_theta_h = fabs(dot(n, w_h));
          float d = gtr_2(cos_theta_h, alpha);
          return d * cos_theta_h / (4.f * fabs(dot(w_o, w_h)));
        }

        inline
        __rtc_device float gtr_2_transmission_pdf(const vec3f &w_o, const vec3f &w_i, const vec3f &n, float transmission_roughness, float ior)
        {
          float alpha = max(MIN_ALPHA, transmission_roughness * transmission_roughness);

          if (same_hemisphere(w_o, w_i, n)) {
            return 0.f;
          }

          // return mat.base_color * abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;


          float eta_o, eta_i;
          bool entering = relative_ior(w_o, n, ior, eta_o, eta_i);
	
          // From Eq 16 of Microfacet models for refraction
          vec3f w_ht = -(w_o * eta_i + w_i * eta_o);
          w_ht = normalize(w_ht);

          float cos_theta_h = fabs(dot(n, w_ht));
          float d = gtr_2(cos_theta_h, alpha);

          return d;// /** cos_theta_h*/ / (4.f * fabs(dot(w_o, w_ht)));

          // vec3f w_r = refract(-w_o, (entering) ? w_ht : -w_ht, eta_o / eta_i);

          // // vec3f n_ = (dot(w_ht, n) > 0) ? n : -;

          // float d = D(w_ht, n, alpha); 
          // float f = F(w_i, w_ht, eta_o, eta_i);
          // float g = G(w_i, w_o, w_ht, alpha); 

          // float i_dot_h = fabs(dot(w_i, w_ht));
          // float o_dot_h = fabs(dot(w_o, w_ht));
          // float i_dot_n = fabs(dot(w_i, n));
          // float o_dot_n = fabs(dot(w_o, n));

          // if (o_dot_n == 0.f || i_dot_n == 0.f) {
          // 	return make_vec3f(0.f);
          // }

          // // From Eq 21 of Microfacet models for refraction
          // float c = (fabs(i_dot_h) * fabs(o_dot_h)) / (fabs(i_dot_n) * fabs(o_dot_n));
          // c *= pow2(eta_o) / pow2(eta_i * i_dot_h + eta_o * o_dot_h);

          // //// hacking in a spherical gaussian here... Can't seem to get microfacet model working...
          // return mat.base_color * abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;

          // return 1.f;
        }

        // __rtc_device float gtr_2_transmission_pdf(const vec3f &w_o, const vec3f &w_i, const vec3f &n,
        // 	float alpha, float ior)
        // {
        // 	if (same_hemisphere(w_o, w_i, n)) {
        // 		return 0.f;
        // 	}
        // 	float eta_o, eta_i;
        // 	relative_ior(w_o, n, ior, eta_i, eta_o);

        // 	// From Eq 16 of Microfacet models for refraction
        // 	vec3f w_h = -(w_o * eta_o + w_i * eta_i);
        // 	w_h = normalize(w_h);

        // 	// // vec3f w_h = normalize(w_i + w_o);
        // 	// float cos_theta_h = fabs(dot(n, w_h));
        // 	// float d = gtr_2(cos_theta_h, alpha);
        // 	// return d * cos_theta_h / (4.f * fabs(dot(w_o, w_h)));


        // 	// vec3f w_h = normalize(w_o + w_i * eta_i / eta_o);
        // 	float cos_theta_h = fabs(dot(n, w_h));
        // 	float i_dot_h = dot(w_i, w_h);
        // 	float o_dot_h = dot(w_o, w_h);
        // 	float d = gtr_2(cos_theta_h, alpha);
        // 	float dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
        // 	return d * cos_theta_h * fabs(dwh_dwi);
        // }

        inline
        __rtc_device float gtr_2_aniso_pdf(const vec3f &w_o, const vec3f &w_i, const vec3f &w_h, const vec3f &n,
                                         const vec3f &v_x, const vec3f &v_y, const vec2f alpha)
        {
          if (!same_hemisphere(w_o, w_i, n)) {
            return 0.f;
          }
          float cos_theta_h = dot(n, w_h);
          float d = gtr_2_aniso(cos_theta_h, fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
          return d * cos_theta_h / (4.f * dot(w_o, w_h));
        }

        inline __rtc_device
        vec3f disney_diffuse_color(const DisneyMaterial &mat,
                                   const vec3f &n,
                                   const vec3f &w_o,
                                   const vec3f &w_i,
                                   const vec3f &w_h)
        {
          return mat.base_color;
        }

        inline
        __rtc_device vec3f disney_subsurface_color(const DisneyMaterial &mat, const vec3f &n,
                                                  const vec3f &w_o, const vec3f &w_i)
        {
          return mat.subsurface_color;
        }

        inline
        __rtc_device void disney_diffuse(const DisneyMaterial &mat, const vec3f &n,
                                       const vec3f &w_o, const vec3f &w_i, const vec3f &w_h, vec3f &bsdf, vec3f &color)
        {
          float n_dot_o = fabs(dot(w_o, n));
          float n_dot_i = fabs(dot(w_i, n));
          float i_dot_h = dot(w_i, w_h);
          float fd90 = 0.5f + 1.0f * mat.roughness * i_dot_h * i_dot_h;
          float fi = schlick_weight(n_dot_i);
          float fo = schlick_weight(n_dot_o);
          color = disney_diffuse_color(mat, n, w_o, w_i, w_h);
          bsdf = (float)M_1_PI * lerp_r(1.f, fd90, fi) * lerp_r(1.f, fd90, fo);
        }

        inline
        __rtc_device void disney_subsurface(const DisneyMaterial &mat, const vec3f &n,
                                          const vec3f &w_o, const vec3f &w_i, const vec3f &w_h, vec3f &bsdf, vec3f &color) {
          float n_dot_o = fabs(dot(w_o, n));
          float n_dot_i = fabs(dot(w_i, n));
          float i_dot_h = dot(w_i, w_h);

          float FL = schlick_weight(n_dot_i), FV = schlick_weight(n_dot_o);
          float Fss90 = i_dot_h*i_dot_h * mat.roughness;
          float Fss = lerp_r(1.0f, Fss90, FL) * lerp_r(1.0f, Fss90, FV);
          float ss = 1.25f * (Fss * (1.f / (n_dot_i + n_dot_o) - .5f) + .5f);
          color = disney_subsurface_color(mat, n, w_o, w_i);
          bsdf = (float)M_1_PI * ss;
        }

        // Eavg in the algorithm is fitted into this
        inline
        __rtc_device float AverageEnergy(float rough){
          float smoothness = 1.f - rough;
          float r = -0.0761947f - 0.383026f * smoothness;
          r = 1.04997f + smoothness * r;
          r = 0.409255f + smoothness * r;
          return min(0.9f, r); 
        }

        // multiple scattering...
        // Favg in the algorithm is fitted into this
        inline
        __rtc_device vec3f AverageFresnel(vec3f specularColor){
          return specularColor + (vec3f(1.f) - specularColor) * (1.0f / 21.0f);
        }

#if 0
        inline __rtc_device vec3f
        disney_multiscatter(const DisneyMaterial &mat,
                            const vec3f &n,
                            const vec3f &w_o,
                            const vec3f &w_i,
                            const vec3f &w_h,
                            rtc::device::TextureObject GGX_E_LOOKUP,
                            rtc::device::TextureObject GGX_E_AVG_LOOKUP)
        {
          float v_dot_n = abs(dot(w_o, n));
          float l_dot_n = abs(dot(w_i, n));
          float alpha = max(mat.roughness*mat.roughness, MIN_ALPHA);

          //E(μ) is in fact the sum of the red and green channels in our environment BRDF
          // vec2 sampleE_o = texture2D(BRDFlut, vec2f(NdotV, alpha)).xy;
          // E_o = sampleE_o.x + sampleE_o.y;
          float E_o = rtc::tex2D<float>(GGX_E_LOOKUP, v_dot_n, alpha);
          float oneMinusE_o = 1.0 - E_o;
          float E_i = rtc::tex2D<float>(GGX_E_LOOKUP, l_dot_n, alpha);
          float oneMinusE_i = 1.0 - E_i;

          float Eavg = rtc::tex2D<float>(GGX_E_AVG_LOOKUP, alpha, .5);
          // float Eavg = AverageEnergy(alpha);
          float oneMinusEavg = 1.0 - Eavg;
	
          float lum = luminance(mat.base_color);
          vec3f tint = lum > 0.f ? mat.base_color / lum : vec3f(1.f);
          vec3f F0 = lerp_r(mat.specular * 0.08f * lerp_r(vec3f(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);
          vec3f Favg = AverageFresnel(F0);

          float brdf = (oneMinusE_o * oneMinusE_i) / (M_PI * oneMinusEavg);
          vec3f energyScale = (Favg * Favg * Eavg) / (vec3f(1.f) - Favg * oneMinusEavg);

          return brdf * energyScale;
        }
#endif

        // __rtc_device float G(vec3f i, vec3f o, vec3f h, float alpha)
        // {
        // 	alpha = 1.f - alpha;
        // 	// Roughly follows Eq 23 from Microfacet Models for Refraction.
        // 	// G is approximately the seperable product of two monodirectional shadowing terms G1 (aka smith shadowing function)
        // 	return smith_shadowing_ggx(fabs(dot(i, h)), alpha) * smith_shadowing_ggx(fabs(dot(o, h)), alpha);
        // }

        // __rtc_device float D(vec3f m, vec3f n, float alpha)
        // {
        // 	// alpha = 1.f - alpha;
        // 	// float alpha_sqr = alpha * alpha;

        // 	// From Eq 33 of Microfacet Models for Refraction
        // 	// help from http://filmicworlds.com/blog/optimizing-ggx-shaders-with-dotlh/
        // 	float m_dot_n = dot(m, n);
        // 	float posCharFunc = (m_dot_n > 0) ? 1.f : 0.f;
        // 	float alpha_sqr = pow2(alpha);
        // 	float denom = m_dot_n * m_dot_n * (alpha_sqr - 1.f) + 1.f;
        // 	float D = alpha_sqr / (M_PI * denom * denom);
        // 	return D;
        // 	// M_1_PI * alpha_sqr / max(pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h), SMALL_EPSILON);
        // 	// return gtr_2(fabs(dot(n, w_ht)), alpha);
        // }

        // __rtc_device float F(vec3f i, vec3f m, float eta_t, float eta_i)
        // {
        // 	// From Eq 22 of Microfacet Models for Refraction
        // 	float c = fabs(dot(i, m));
        // 	float g = pow2(eta_t) / pow2(eta_i) - 1 + pow2(c);
        // 	if (g < 0) return 1; // if g is imaginary after sqrt, this indicates a total internal reflection
        // 	g = sqrtf(g);
        // 	float f = .5f;
        // 	f *= pow2(g - c) / pow2(g + c);
        // 	f *= (1.f + (pow2(c * (g + c) - 1) / pow2(c * (g - c) + 1)));
        // 	return f;
        // }

        inline __rtc_device
        vec3f disney_microfacet_reflection_color(const DisneyMaterial &mat,
                                                  const vec3f &n,
                                                  
                                                  const vec3f &w_o,
                                                  const vec3f &w_i,
                                                  const vec3f &w_h)
        {
          float lum = luminance(mat.base_color);
          vec3f tint = lum > 0.f ? mat.base_color / lum : vec3f(1.f);
          vec3f spec = lerp_r(mat.specular * 0.08f * lerp_r(vec3f(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

          float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
          vec3f f = lerp_r(spec, vec3f(1.f), schlick_weight(dot(w_o, n)));
          return f;
        }

        inline
        __rtc_device vec3f
        disney_microfacet_isotropic(const DisneyMaterial &mat,
                                    const vec3f &n,
                                    const vec3f &w_o,
                                    const vec3f &w_i,
                                    const vec3f &w_h,
                                    bool dbg = false)
        {
          float lum = luminance(mat.base_color);
          vec3f tint = lum > 0.f ? mat.base_color / lum : vec3f(1.f);
          vec3f spec = lerp_r(mat.specular * 0.08f
                              * lerp_r(vec3f(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

          float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
          float d = gtr_2(fabsf(dot(n, w_h)), alpha, dbg);
          // Finding dot(w_o, n) to be less noisy, but doesn't look as good for crazy normal maps compared to dot(w_i, w_h)
          // Also finding fresnel to be introducing unwanted energy for smooth plastics, so I'm adding a correction term.
          vec3f f = lerp_r(spec, vec3f(1.f),
                            schlick_weight(fabs(dot(w_i, w_h)))
                            * lerp_r(.5f, 1.f, max(mat.metallic, alpha))
                            );
          float g_i = smith_shadowing_ggx(fabs(dot(n, w_i)), alpha);
          float g_o = smith_shadowing_ggx(fabs(dot(n, w_o)), alpha);
          float g   = g_i * g_o;
          if (dbg)
            printf("microfacet_iso spec %f %f %f d %f f %f %f %f g %f (%f %f)\n",
                    spec.x,spec.y,spec.z,
                    d,f.x,f.y,f.z,g,g_i,g_o);
          return d * f * g;
        }

        inline
        __rtc_device vec3f disney_microfacet_transmission_color(const DisneyMaterial &mat, const vec3f &n,
                                                               const vec3f &w_o, const vec3f &w_i, const vec3f &w_h)
        {	
          // Approximate absorption
          // note that compositing will be incorrect...
          return mat.base_color;
        }

        inline
        __rtc_device void
        disney_microfacet_transmission_isotropic(const DisneyMaterial &mat,
                                                 const vec3f &n,

                                                 const vec3f &w_o,
                                                 const vec3f &w_i,
                                                 float &bsdf,
                                                 vec3f &color)
        {	

          float eta_o, eta_i;
          bool entering = relative_ior(w_o, n, mat.ior, eta_o, eta_i);

          float alpha = max(MIN_ALPHA, mat.transmission_roughness * mat.transmission_roughness);
	
          // From Eq 16 of Microfacet models for refraction
          vec3f w_ht = -(w_o * eta_i + w_i * eta_o);
          w_ht = normalize(w_ht);

          color = disney_microfacet_transmission_color(mat, n, w_o, w_i, w_ht);

          float lum = luminance(color);
          vec3f tint = lum > 0.f ? color / lum : vec3f(1.f);
	
          vec3f spec = disney_microfacet_transmission_color(mat, n, w_o, w_i, w_ht);

          // float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
          float cos_theta_h = fabs(dot(n, w_ht));
          float d = gtr_2(cos_theta_h, alpha);
          vec3f f = lerp_r(spec, vec3f(1.f), 1.0f - schlick_weight(dot(w_i, w_ht)));
          float g = smith_shadowing_ggx(abs(dot(n, w_i)), alpha) * smith_shadowing_ggx(abs(dot(n, w_o)), alpha);
	
          bsdf = d;
          color = spec;
          return;// * f;// * g; // * f * g;



          // // w_ht = n; // HACK

          // vec3f w_r = refract(-w_o, (entering) ? w_ht : -w_ht, eta_o / eta_i);

          // // vec3f n_ = (dot(w_ht, n) > 0) ? n : -;

	

          // // float d = gtr_2();//D(w_ht, n, alpha); 
          // float f = F(w_i, w_ht, eta_o, eta_i);
          // float g = G(w_i, w_o, w_ht, alpha); 

          // float i_dot_h = fabs(dot(w_i, w_ht));
          // float o_dot_h = fabs(dot(w_o, w_ht));
          // float i_dot_n = fabs(dot(w_i, n));
          // float o_dot_n = fabs(dot(w_o, n));

          // if (o_dot_n == 0.f || i_dot_n == 0.f) {
          // 	return make_vec3f(0.f);
          // }

          // // From Eq 21 of Microfacet models for refraction
          // float c = (fabs(i_dot_h) * fabs(o_dot_h)) / (fabs(i_dot_n) * fabs(o_dot_n));
          // c *= pow2(eta_o) / pow2(eta_i * i_dot_h + eta_o * o_dot_h);

          // //// hacking in a spherical gaussian here... Can't seem to get microfacet model working...
          // return mat.base_color * d;// * f * g; //abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;
        }

        inline
        __rtc_device vec3f disney_microfacet_anisotropic(const DisneyMaterial &mat, const vec3f &n,
                                                        const vec3f &w_o, const vec3f &w_i, const vec3f &w_h, const vec3f &v_x, const vec3f &v_y)
        {
          float lum = luminance(mat.base_color);
          vec3f tint = lum > 0.f ? mat.base_color / lum : vec3f(1.f);
          vec3f spec = lerp_r(mat.specular * 0.08f * lerp_r(vec3f(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

          float aspect = sqrtf(1.f - mat.anisotropy * 0.9f);
          float a = max(MIN_ALPHA,mat.roughness * mat.roughness);
          vec2f alpha = vec2f(max(MIN_ALPHA, a / aspect), max(MIN_ALPHA, a * aspect));
          float d = gtr_2_aniso(fabs(dot(n, w_h)), fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
          // Finding dot(w_o, n) to be less noisy, but doesn't look as good for crazy normal maps compared to dot(w_i, w_h)
          // Also finding fresnel to be introducing unwanted energy for smooth plastics, so I'm adding a correction term.
          vec3f f = lerp_r(spec, vec3f(1.f), schlick_weight(fabs(dot(w_i, w_h))) * lerp_r(.5f, 1.f, max(mat.metallic, alpha.x * alpha.y)));
          float g = smith_shadowing_ggx_aniso(fabs(dot(n, w_i)), fabs(dot(w_i, v_x)), fabs(dot(w_i, v_y)), alpha)
            * smith_shadowing_ggx_aniso(fabs(dot(n, w_o)), fabs(dot(w_o, v_x)), fabs(dot(w_o, v_y)), alpha);
          return d * f * g;
        }

        inline
        __rtc_device float disney_clear_coat(const DisneyMaterial &mat, const vec3f &n,
                                           const vec3f &w_o, const vec3f &w_i, const vec3f &w_h)
        {
          float alpha = lerp_r(0.1f, MIN_ALPHA, mat.clearcoat_gloss);
          float d = gtr_1(fabs(dot(n, w_h)), alpha);
          float f = lerp_r(0.04f, 1.f, schlick_weight(dot(w_i, n)));
          float g = smith_shadowing_ggx(fabs(dot(n, w_i)), 0.25f) * smith_shadowing_ggx(fabs(dot(n, w_o)), 0.25f);
          return /*0.25f * */mat.clearcoat * d * f * g;
        }

        inline
        __rtc_device vec3f disney_sheen(const DisneyMaterial &mat, const vec3f &n,
                                       const vec3f &w_o, const vec3f &w_i, const vec3f &w_h)
        {
          float lum = luminance(mat.base_color);
          vec3f tint = lum > 0.f ? mat.base_color / lum : vec3f(1.f);
          vec3f sheen_color = lerp_r(vec3f(1.f), tint, mat.sheen_tint);
          float f = schlick_weight(dot(w_i, n));
          return f * mat.sheen * sheen_color;
        }

        /* 
         * Compute the throughput of a given sampled direction
         * @param mat The structure containing material information.
         * @param g_n The geometric normal (cross product of the two triangle edges)
         * @param s_n The shading normal (per-vertex interpolated normal)
         * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
         * @param v_x The tangent vector
         * @param v_y The binormal vector
         * @param w_o The outgoing (aka view) vector
         * @param w_i The sampled incoming (aka light) vector
         * @param w_h The halfway vector between the incoming and outgoing vectors
         * @param pdf The returned probability of this sample
         */
        inline
        __rtc_device void disney_brdf(
                                    const DisneyMaterial &mat, 
                                    const vec3f &g_n,
                                    const vec3f &s_n,
                                    const vec3f &b_n,
                                    const vec3f &v_x, 
                                    const vec3f &v_y,
                                    const vec3f &w_o, 
                                    const vec3f &w_i, 
                                    const vec3f &w_h, 
                                    vec3f &bsdf,
                                    bool dbg = false
                                    ) {
          // initialize bsdf value to black for now.

          bsdf = vec3f(0.f);

          // transmissive objects refract when back of surface is visible.
          if (!same_hemisphere(w_o, w_i, b_n) && (mat.specular_transmission > 0.f)) {
            float spec_trans; vec3f trans_color;
            disney_microfacet_transmission_isotropic(mat, b_n, w_o, w_i, spec_trans, trans_color);
            spec_trans = spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
            bsdf = vec3f(spec_trans) * trans_color;			
            return;
          }

          float coat = disney_clear_coat(mat, b_n, w_o, w_i, w_h);
          vec3f sheen = disney_sheen(mat, b_n, w_o, w_i, w_h);
          vec3f diffuse_bsdf, diffuse_color;
          disney_diffuse(mat, b_n, w_o, w_i, w_h, diffuse_bsdf, diffuse_color);
          vec3f subsurface_bsdf, subsurface_color;
          disney_subsurface(mat, b_n, w_o, w_i, w_h, subsurface_bsdf, subsurface_color);
          vec3f gloss;
          if (mat.anisotropy == 0.f) {
            gloss =
              disney_microfacet_isotropic(mat, b_n, w_o, w_i, w_h, dbg);
            // gloss = gloss + disney_multiscatter(mat, n, w_o, w_i, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
          } else {
            gloss = disney_microfacet_anisotropic(mat, b_n, w_o, w_i, w_h, v_x, v_y);
              // gloss = gloss + disney_multiscatter(mat, n, w_o, w_i, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
          }
          
	
          if (dbg) printf("nvis gloss %f %f %f\n",gloss.x,gloss.y,gloss.z);
          if (dbg) printf("nvis diffuse bsdf %f %f %f color %f %f %f, (1-metal)*(1-spec) %f\n",
                          diffuse_bsdf.x,
                          diffuse_bsdf.y,
                          diffuse_bsdf.z,
                          diffuse_color.x,
                          diffuse_color.y,
                          diffuse_color.z,
                          (1.f - mat.metallic) * (1.f - mat.specular_transmission)
                          );

          vec3f flat = lerp_r(diffuse_bsdf * diffuse_color, 
                               subsurface_bsdf * subsurface_color, 
                               mat.flatness);
          if (dbg) printf("BRDF: flat %f %f %f\n",flat.x,flat.y,flat.z);
          if (dbg) printf("BRDF: 1-met %f 1-spec %f sheen %f %f %f coat %f gloss %f %f %f aniso %f\n",1.f-mat.metallic,1.f-mat.specular_transmission,
                          sheen.x,sheen.y,sheen.z,
                          coat,
                          gloss.x,gloss.y,gloss.z,
                           mat.anisotropy);
          
          
          bsdf = (flat
                  * (1.f - mat.metallic) * (1.f - mat.specular_transmission)
                  + sheen + coat + gloss);// * fabs(dot(w_i, b_n));
        }

        /* 
         * Compute the probability of a given sampled direction
         * @param mat The structure containing material information.
         * @param g_n The geometric normal (cross product of the two triangle edges)
         * @param s_n The shading normal (per-vertex interpolated normal)
         * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
         * @param v_x The tangent vector
         * @param v_y The binormal vector
         * @param w_o The outgoing (aka view) vector
         * @param w_i The sampled incoming (aka light) vector
         * @param w_h The halfway vector between the incoming and outgoing vectors
         * @param pdf The returned probability of this sample
         */
        inline
        __rtc_device void disney_pdf(
                                   const DisneyMaterial &mat, 
                                   const vec3f &g_n,
                                   const vec3f &s_n,
                                   const vec3f &b_n,
                                   const vec3f &v_x, 
                                   const vec3f &v_y,
                                   const vec3f &w_o, 
                                   const vec3f &w_i, 
                                   const vec3f &w_h, 
                                   float &pdf,
                                   bool dbg)
        {
#if IMPORTANCE_SAMPLE_BRDF
          float diffuse_weight
            = //.01f +
            (1.f - mat.metallic) * (1.f - mat.specular_transmission);
          float glossy_weight
            = 1.f+mat.metallic + mat.sheen + mat.specular + mat.roughness;
          
          // = mat.metallic + mat.specular_transmission + mat.clearcoat;
          // * (1.f - mat.metallic) * (1.f - mat.specular_transmission) 
          //           + sheen + coat + gloss) * fabs(dot(w_i, b_n));
          float clearcoat_weight// = (1.f-mat.metallic)*mat.clearcoat;
            = mat.clearcoat;
          float transmission_weight
            = 0.f;
            // = (1.f-mat.metallic)*mat.specular_transmission
            // * (dot(w_o, b_n) > 0.f);
          float sum_weights = diffuse_weight+glossy_weight
            +clearcoat_weight+transmission_weight;
          if (sum_weights == 0.f) {
            // printf("no importance sampling weights...\n");
            pdf = 0.f;
            return;
          }
                   
          float scale_weights
            = 1.f
            // /sum_weights
            ;
          diffuse_weight      *= scale_weights;
          glossy_weight       *= scale_weights;
          clearcoat_weight    *= scale_weights;
          transmission_weight *= scale_weights;
#endif
          pdf = 0.f;

          bool entering = dot(w_o, b_n) > 0.f;
          bool sameHemisphere = same_hemisphere(w_o, w_i, b_n);
	
          float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
          float t_alpha = max(MIN_ALPHA, mat.transmission_roughness * mat.transmission_roughness);
          float aspect = sqrtf(1.f - mat.anisotropy * 0.9f);
          vec2f alpha_aniso = vec2f(max(MIN_ALPHA, alpha / aspect), max(MIN_ALPHA, alpha * aspect));

          float clearcoat_alpha = lerp_r(0.1f, MIN_ALPHA, mat.clearcoat_gloss);

          float diffuse = lambertian_pdf(w_i, b_n);
          float clear_coat
            = //mat.clearcoat *
            gtr_1_pdf(w_o, w_i, w_h, b_n, clearcoat_alpha);

#if IMPORTANCE_SAMPLE_BRDF
#else
          float n_comp = 3.f;
#endif
          float microfacet = 0.f;
          float microfacet_transmission = 0.f;
          if (mat.anisotropy == 0.f) {
            microfacet = gtr_2_pdf(w_o, w_i, w_h, b_n, alpha);
          } else {
            microfacet = gtr_2_aniso_pdf(w_o, w_i, w_h, b_n, v_x, v_y, alpha_aniso);
          }

          if ((mat.specular_transmission > 0.f) &&
              (!same_hemisphere(w_o, w_i, b_n))) {
            microfacet_transmission
              = gtr_2_transmission_pdf(w_o, w_i, b_n, mat.transmission_roughness, mat.ior);
          } 

          // not sure why, but energy seems to be added from smooth metallic. By subtracting mat.metallic from n_comps,
          // we decrease brightness and become almost perfectly conserving energy for shiny metallic. As metals get 
          // rough, we lose energy from our single scattering microfacet model around .1 roughness, so we 
          // remove the energy reduction kludge for metallic in the case of slightly rough metal. We still 
          // seem to lose a lot of energy in that case, and could likely benefit from a multiple scattering microfacet
          // model. 
          // For transmission, so long as we subtract 1 from the components, we seem to preserve energy
          // regardless if the transmission is rough or smooth.
          
          // float metallic_kludge = mat.metallic;
          // float transmission_kludge = mat.specular_transmission;
          // n_comp -= lerp_r(transmission_kludge, metallic_kludge, mat.metallic);

#if IMPORTANCE_SAMPLE_BRDF
          // if (dbg) printf("PDF diff %f micro %f trans %f coat %f\n",
          //                 diffuse,microfacet,microfacet_transmission,clear_coat);
          pdf = (diffuse_weight*diffuse
                 + glossy_weight*microfacet
                 + clearcoat_weight*clear_coat
                 + transmission_weight*microfacet_transmission);
#else
          pdf = (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
#endif
          // if (dbg) printf(" nvis pdf diffuse %f microfacet %f clarcoat %f -> pdf %f\n",
          //                 diffuse,microfacet,clear_coat);
        }

        /* 
         * Sample a component of the Disney BRDF
         * @param mat The structure containing material information.
         * @param rng The random number generator
         * @param g_n The geometric normal (cross product of the two triangle edges)
         * @param s_n The shading normal (per-vertex interpolated normal)
         * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
         * @param v_x The tangent vector
         * @param v_y The binormal vector
         * @param w_o The outgoing (aka view) vector
         * @param w_i The returned incoming (aka light) vector
         * @param pdf The probability of this sample, for importance sampling
         * @param sampled_bsdf Enum for which bsdf was sampled. 
         * 	Can be either DISNEY_DIFFUSE_BRDF, DISNEY_GLOSSY_BRDF, DISNEY_CLEARCOAT_BRDF, DISNEY_TRANSMISSION_BRDF
         * @param bsdf The throughput of all brdfs in the sampled direction
         */
        inline
        __rtc_device void sample_disney_brdf(
                                           const DisneyMaterial &mat,
                                           Random &rng,
                                           const vec3f &g_n, const vec3f &s_n, const vec3f &b_n, 
                                           const vec3f &v_x, const vec3f &v_y,
                                           const vec3f &w_o,
                                           vec3f &w_i, 
                                           float &pdf, 
                                           int &sampled_bsdf, 
                                           vec3f &bsdf,
                                           bool dbg
                                           ) {
#if IMPORTANCE_SAMPLE_BRDF
          // vec3f base_color;
          // vec3f subsurface_color;
          // float metallic;

          // float specular;
          // float roughness;
          // float specular_tint;
          // float anisotropy;

          // float sheen;
          // float sheen_tint;
          // float clearcoat;
          // float clearcoat_gloss;

          // float ior;
          // float specular_transmission;
          // float transmission_roughness;
          // float flatness;
          // float alpha;
          // iw - use importance sampling
          float diffuse_weight
            = //.01f +
            (1.f - mat.metallic) * (1.f - mat.specular_transmission);
          float glossy_weight
            = 1.f+mat.metallic + mat.sheen + mat.specular + mat.roughness;
          
          // = mat.metallic + mat.specular_transmission + mat.clearcoat;
          // * (1.f - mat.metallic) * (1.f - mat.specular_transmission) 
          //           + sheen + coat + gloss) * fabs(dot(w_i, b_n));
          float clearcoat_weight// = (1.f-mat.metallic)*mat.clearcoat;
            = mat.clearcoat;
          float transmission_weight = 0.f;
            // = (1.f-mat.metallic)*mat.specular_transmission
            // * (dot(w_o, b_n) > 0.f);
          float sum_weights = diffuse_weight+glossy_weight
            +clearcoat_weight+transmission_weight;
          if (sum_weights == 0.f) {
            printf("no importance sampling weights...\n");
            pdf = 0.f;
            return;
          }
                   
          float scale_weights
            = 1.f/sum_weights;
          diffuse_weight      *= scale_weights;
          glossy_weight       *= scale_weights;
          clearcoat_weight    *= scale_weights;
          transmission_weight *= scale_weights;
          // if (dbg) printf("scatter type weights %f %f %f %f\n",
          //                 diffuse_weight,glossy_weight,clearcoat_weight,transmission_w
          // eight);
          float type_rng = lcg_randomf(rng);
          float type_pdf = 0.f;
          if (type_rng < diffuse_weight) {
            type_pdf = diffuse_weight;
            sampled_bsdf = DISNEY_DIFFUSE_BRDF;
          } else if ((type_rng-diffuse_weight) < glossy_weight) {
            type_pdf = glossy_weight;
            sampled_bsdf = DISNEY_GLOSSY_BRDF;
          } else if ((type_rng-diffuse_weight-glossy_weight) < clearcoat_weight) {
            type_pdf = clearcoat_weight;
            sampled_bsdf = DISNEY_CLEARCOAT_BRDF;
          } else {
            type_pdf = transmission_weight;
            sampled_bsdf = DISNEY_TRANSMISSION_BRDF;
          }
          type_pdf = type_pdf;
          // type_pdf *= 3.f;
#else
          const float type_pdf = 1.f;
            // Randomly pick a brdf to sample
          if (mat.specular_transmission == 0.f) {
            if (dbg) printf(" => scatter(1)\n");
            sampled_bsdf = lcg_randomf(rng) * 3.f;
            sampled_bsdf = clamp(sampled_bsdf, 0, 2);
          } else {
            // If we're looking at the front face 
            if (dot(w_o, b_n) > 0.f) {
              sampled_bsdf = lcg_randomf(rng) * 4.f;
              sampled_bsdf = clamp(sampled_bsdf, 0, 3);
              // if (dbg) printf("-> sampled %i\n",sampled_bsdf);
            }
            else sampled_bsdf = DISNEY_TRANSMISSION_BRDF; 
          }
#endif
          if (dbg) printf(" => scatter type %i\n",sampled_bsdf);

          vec2f samples = vec2f(lcg_randomf(rng), lcg_randomf(rng));
          if (sampled_bsdf == DISNEY_DIFFUSE_BRDF) {
            w_i = sample_lambertian_dir(b_n, v_x, v_y, samples);
          } else if (sampled_bsdf == DISNEY_GLOSSY_BRDF) {
            vec3f w_h;
            float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
            if (mat.anisotropy == 0.f) {
              w_h = sample_gtr_2_h(b_n, v_x, v_y, alpha, samples);
            } else {
              float aspect = sqrtf(1.f - mat.anisotropy * 0.9f);
              vec2f alpha_aniso = vec2f(max(MIN_ALPHA, alpha / aspect), max(MIN_ALPHA, alpha * aspect));
              w_h = sample_gtr_2_aniso_h(b_n, v_x, v_y, alpha_aniso, samples);
            }
            w_i = reflect(-w_o, w_h);

            // Invalid reflection, terminate ray
            if (!same_hemisphere(w_o, w_i, b_n)) {
              pdf = 0.f;
              w_i = vec3f(0.f);
              bsdf = vec3f(0.f);
              return;
            }
          } else if (sampled_bsdf == DISNEY_CLEARCOAT_BRDF) {
            float alpha = lerp_r(0.1f, MIN_ALPHA, mat.clearcoat_gloss);
            vec3f w_h = sample_gtr_1_h(b_n, v_x, v_y, alpha, samples);
            w_i = reflect(-w_o, w_h);

            // Invalid reflection, terminate ray
            if (!same_hemisphere(w_o, w_i, b_n)) {
              pdf = 0.f;
              w_i = vec3f(0.f);
              bsdf = vec3f(0.f);
              return;
            }
          } else if (sampled_bsdf == DISNEY_TRANSMISSION_BRDF) {	
            float alpha = max(MIN_ALPHA, mat.transmission_roughness * mat.transmission_roughness);
            vec3f w_h = sample_gtr_2_h(b_n, v_x, v_y, alpha, samples);
            float eta_o, eta_i;
            bool entering = relative_ior(w_o, w_h, mat.ior, eta_o, eta_i);
            // w_i = refract(-w_o, w_h, eta_o / eta_i);

            // w_o is flipped, so we also flip eta
            w_i = refract(-w_o, (entering) ? w_h : -w_h, eta_i / eta_o);

            // Total internal reflection
            if (all_zero(w_i)) {
              w_i = reflect(-w_o, (entering) ? w_h : -w_h);
              pdf = 1.f;
              bsdf = vec3f(1.f);// Normally absorption would happen here...
              return;
            }
          }
	
          vec3f w_h = normalize(w_i + w_o);
          disney_pdf(mat, g_n, s_n, b_n, v_x, v_y, w_o, w_i, w_h, pdf, dbg);
          if (dbg) printf("-> got pdf %f\n",pdf);

// #if 1
          // pdf *= type_pdf / 4.f;
          // pdf *= type_pdf;
// #endif
          
          if (dbg) printf("-> is-adjusted pdf %f\n",pdf);
          disney_brdf(mat, g_n, s_n, b_n, v_x, v_y, w_o, w_i, w_h, bsdf, dbg);
          if (dbg) printf("-> got bsdf %f %f %f\n",bsdf.x,bsdf.y,bsdf.z);
        }
      }
      
      struct NVisii {
        inline __rtc_device
        nvisii::DisneyMaterial unpack() const {
          nvisii::DisneyMaterial mat;
          mat.base_color = (vec3f)baseColor;
          mat.subsurface_color = (vec3f)subsurfaceColor;
          mat.metallic  = metallic;
          mat.specular  = specular;
          mat.roughness = roughness;
          mat.specular_tint = specularTint;
          mat.anisotropy = anisotropy;
          mat.sheen = sheen;
          mat.sheen_tint = sheenTint;
          mat.clearcoat = clearcoat;
          mat.clearcoat_gloss = clearcoatGloss;
          mat.ior = ior;
          mat.specular_transmission = specularTransmission;
          mat.transmission_roughness = transmissionRoughness;
          mat.flatness = flatness;
          mat.alpha = alpha;
          return mat;
        }
        inline __rtc_device vec3f getAlbedo(bool dbg) const;
        inline __rtc_device
        float getOpacity(bool isShadowRay,
                         bool isInMedium,
                         vec3f rayDir,
                         vec3f Ng,
                         bool dbg=false) const
        {
          return (float)alpha;
        }
        inline __rtc_device EvalRes eval(DG dg, vec3f wi, bool dbg) const;
        inline __rtc_device float pdf(DG dg, vec3f wi, bool dbg) const;
        inline __rtc_device void scatter(ScatterResult &scatter,
                                       const render::DG &dg,
                                       Random &random,
                                       bool dbg) const;
        inline __rtc_device void setDefaults()
        {
        //   	this->base_color = vec4(.8, .8, .8, 1.0);
	// this->subsurface_radius = vec4(1.0, .2, .1, 1.0);
	// this->subsurface_color = vec4(.8, .8, .8, 1.0);
	// this->subsurface = 0.0;
	// this->metallic = 0.0;
	// this->specular = .5;
	// this->specular_tint = 0.0;
	// this->roughness = .5;
	// this->anisotropic = 0.0;
	// this->anisotropic_rotation = 0.0;
	// this->sheen = 0.0;
	// this->sheen_tint = 0.5;
	// this->clearcoat = 0.0;
	// this->clearcoat_roughness = .03f;
	// this->ior = 1.45f;
	// this->transmission = 0.0;
	// this->transmission_roughness = 0.0;

          this->baseColor = vec3f(.8f);
          this->subsurfaceColor = vec3f(.8f);
          this->metallic = 0.f;

          this->specular = .5f;
          this->roughness = .5f;
          this->specularTint = 0.f;
          this->anisotropy = 0.f;

          this->sheen = 0.f;
          this->sheenTint = .5f;
          this->clearcoat = 0.f;
          // nate
          float clearcoat_roughness = .03f;
          this->clearcoatGloss = 1.f - clearcoat_roughness*clearcoat_roughness;
          this->ior = 1.45f;
          this->specularTransmission = 0.f;
          const float MIN_ROUGHNESS = .04f;
          this->transmissionRoughness = MIN_ROUGHNESS;
          this->flatness = 0.f;
          this->alpha = 1.f;
        }
        
        
	vec3h baseColor;
	vec3h subsurfaceColor;
	half metallic;

	half specular;
	half roughness;
	half specularTint;
	half anisotropy;

	half sheen;
	half sheenTint;
	half clearcoat;
	half clearcoatGloss;

	half ior;
	half specularTransmission;
	half transmissionRoughness;
	half flatness;
	half alpha;
      };

      inline __rtc_device vec3f NVisii::getAlbedo(bool dbg) const
      {
        vec3f baseColor = this->baseColor;
        // if (dbg) printf("visrtx::getalbedo %f %f %f\n",
        //                 (float)baseColor.x,
        //                 (float)baseColor.y,
        //                 (float)baseColor.z); 
        return baseColor;
      }

      inline __rtc_device void NVisii::scatter(ScatterResult &scatter,
                                             const render::DG &dg,
                                             Random &rng,
                                             bool dbg) const
      {
// #if 1
        /* 
         * Sample a component of the Disney BRDF
         * @param mat The structure containing material information.
         * @param rng The random number generator
         * @param g_n The geometric normal (cross product of the two triangle edges)
         * @param s_n The shading normal (per-vertex interpolated normal)
         * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
         * @param v_x The tangent vector
         * @param v_y The binormal vector
         * @param w_o The outgoing (aka view) vector
         * @param w_i The returned incoming (aka light) vector
         * @param pdf The probability of this sample, for importance sampling
         * @param sampled_bsdf Enum for which bsdf was sampled. 
         * 	Can be either DISNEY_DIFFUSE_BRDF, DISNEY_GLOSSY_BRDF, DISNEY_CLEARCOAT_BRDF, DISNEY_TRANSMISSION_BRDF
         * @param bsdf The throughput of all brdfs in the sampled direction
         */
        // inline
        // __rtc_device void sample_disney_brdf(
        //                                    const DisneyMaterial &mat,
        //                                    LCGRand &rng,
        //                                    const vec3f &g_n, const vec3f &s_n, const vec3f &b_n, 
        //                                    const vec3f &v_x, const vec3f &v_y,
        //                                    const vec3f &w_o,
        //                                    vec3f &w_i, 
        //                                    float &pdf, 
        //                                    int &sampled_bsdf, 
        //                                    vec3f &bsdf
        //                                    ) {
        using namespace nvisii;
        DisneyMaterial mat = unpack();
        mat.alpha = 1.f;

        // * @param g_n The geometric normal (cross product of the two triangle edges)
        vec3f g_n = (vec3f)dg.Ng;
         // * @param s_n The shading normal (per-vertex interpolated normal)
        vec3f s_n = (vec3f)dg.Ns;
         // * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
        vec3f b_n = s_n;
         // * @param w_i The sampled incoming (aka light) vector
        // vec3f w_i = (vec3f)wi;
         // * @param w_o The outgoing (aka view) vector
        vec3f w_o = dg.wo;
         // * @param w_h The halfway vector between the incoming and outgoing vectors
        // vec3f w_h = normalize(w_i+w_o);
         // * @param v_y The binormal vector
        vec3f v_y = normalize(cross(g_n,w_o));
         // * @param v_x The tangent vector
        vec3f v_x = normalize(cross(g_n,v_y));

        // if (isnan(v_x) || isnan(v_y)) {
        //   printf("============================ NAN ==============================\n");
        //   printf("============================ NAN ==============================\n");
        //   printf("============================ NAN ==============================\n");
        //   printf("============================ NAN ==============================\n");
        //   printf("============================ NAN ==============================\n");
        // }
        // out:
        vec3f w_i;
        int    sampled_bsdf;
        float  pdf;
        vec3f bsdf;
        sample_disney_brdf(mat,rng,g_n,s_n,b_n,v_x,v_y,w_o,
                           // out:
                           w_i, pdf, sampled_bsdf, bsdf, dbg);
        if (0 && dbg) printf(" -> nvis sampled type %i dir %f %f %f bsdf %f %f %f pdf %f\n",
                        sampled_bsdf,
                        w_i.x,
                        w_i.y,
                        w_i.z,
                        bsdf.x,
                        bsdf.y,
                        bsdf.z,
                        pdf);
        scatter.pdf = pdf;
        scatter.f_r = bsdf;
        scatter.dir = normalize(w_i);
        scatter.type
          = (sampled_bsdf == DISNEY_DIFFUSE_BRDF)
          ? ScatterResult::DIFFUSE
          : ScatterResult::GLOSSY;

        if (dbg) printf(" => done scatter, f_r %f %f %f pdf %f\n",
                        scatter.f_r.x,
                        scatter.f_r.y,
                        scatter.f_r.z,
                        scatter.pdf
                        );
      }

      inline __rtc_device EvalRes NVisii::eval(DG dg, vec3f wi, bool dbg) const
      {
        using namespace nvisii;
        DisneyMaterial mat = unpack();

        mat.alpha = 1.f;
        // * @param g_n The geometric normal (cross product of the two triangle edges)
        vec3f g_n = (vec3f)dg.Ng;
         // * @param s_n The shading normal (per-vertex interpolated normal)
        vec3f s_n = (vec3f)dg.Ns;
         // * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
        vec3f b_n = s_n;
         // * @param w_i The sampled incoming (aka light) vector
        vec3f w_i = (vec3f)wi;
         // * @param w_o The outgoing (aka view) vector
        vec3f w_o = dg.wo;
         // * @param w_h The halfway vector between the incoming and outgoing vectors
        vec3f w_h = normalize(w_i+w_o);
         // * @param v_y The binormal vector
        vec3f v_y = normalize(cross(w_o,g_n));
         // * @param v_x The tangent vector
        vec3f v_x = normalize(cross(g_n,v_y));
        
        vec3f bsdf;
        disney_brdf(mat, g_n,s_n,b_n,v_x, v_y,w_o,w_i, w_h, bsdf,dbg);
        EvalRes ret;
        if (dbg) printf("nvisii bsdf %f %f %f\n",bsdf.x,bsdf.y,bsdf.z);
        ret.value = vec3f(bsdf);
        disney_pdf(mat, g_n,s_n,b_n,v_x, v_y,w_o,w_i, w_h, ret.pdf,dbg);
        return ret;
      }

      inline __rtc_device float NVisii::pdf(DG dg, vec3f wi, bool dbg) const
      {
        using namespace nvisii;
        DisneyMaterial mat = unpack();
        mat.alpha = 1.f;
        
        vec3f g_n = (vec3f)dg.Ng;
         // * @param s_n The shading normal (per-vertex interpolated normal)
        vec3f s_n = (vec3f)dg.Ns;
         // * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
        vec3f b_n = s_n;
         // * @param w_i The sampled incoming (aka light) vector
        vec3f w_i = (vec3f)wi;
         // * @param w_o The outgoing (aka view) vector
        vec3f w_o = dg.wo;
         // * @param w_h The halfway vector between the incoming and outgoing vectors
        vec3f w_h = normalize(w_i+w_o);
         // * @param v_y The binormal vector
        vec3f v_y = normalize(cross(w_o,g_n));
         // * @param v_x The tangent vector
        vec3f v_x = normalize(cross(g_n,v_y));

        float pdf;
        disney_pdf(mat, g_n,s_n,b_n,v_x, v_y,w_o,w_i, w_h, pdf,dbg);
        return pdf;
      }
    }    
  }
}

