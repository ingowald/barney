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

#include "barney/material/device/DG.h"
#include "barney/material/device/BSDF.h"
#include "barney/material/bsdfs/Lambert.h"

namespace barney {
  namespace nvisii {
    // ==================================================================
    // iw - wrappers....
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#define SMALL_EPSILON 0.00000000001f
#define MIN_ROUGHNESS .04f
#define MIN_ALPHA MIN_ROUGHNESS * MIN_ROUGHNESS

#define DISNEY_DIFFUSE_BRDF 0
#define DISNEY_GLOSSY_BRDF 1
#define DISNEY_CLEARCOAT_BRDF 2
#define DISNEY_TRANSMISSION_BRDF 3

    using LCGRand = barney::Random;
    inline __device__ float lcg_randomf(LCGRand &rng) { return rng(); }
    
    inline __device__ float clamp(float f, float lo, float hi) { return max(lo,min(hi,f)); }
    inline __device__ float sqrt(float f) { return sqrtf(f); }
    inline __device__ float saturate(float f) { return clamp(f,0.f,1.f); }
    inline __device__ float pow2(float x) { return x * x; }


    inline __device__ float2 make_float2(float f) { return ::make_float2(f,f); }
    inline __device__ float2 make_float2(float x, float y) { return ::make_float2(x,y); }
    inline __device__ float2 operator*(float f, float2 v)
    { return make_float2(f*v.x,f*v.y); }
    inline __device__ float2 operator-(float2 a, float2 b)
    { return make_float2(a.x-b.x,a.y-b.y); }
    
    inline __device__ float3 make_float3(float f) { return ::make_float3(f,f,f); }
    inline __device__ float3 make_float3(float x, float y, float z) { return ::make_float3(x,y,z); }
    inline __device__ float dot(float3 a, float3 b)
    { return a.x*b.x+a.y*b.y+a.z*b.z; }
    inline __device__ float3 operator*(float f, float3 v)
    { return make_float3(f*v.x,f*v.y,f*v.z); }
    inline __device__ float3 operator*(float3 v, float f)
    { return make_float3(f*v.x,f*v.y,f*v.z); }
    inline __device__ float3 operator+(float3 v, float f)
    { return make_float3(f+v.x,f+v.y,f+v.z); }
    inline __device__ float3 operator/(float3 v, float f)
    { return make_float3(v.x/f,v.y/f,v.z/f); }
    inline __device__ float3 operator-(float3 v)
    { return make_float3(-v.x,-v.y,-v.z); }
    inline __device__ float3 operator+(float3 a, float3 b)
    { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
    inline __device__ float3 operator/(float3 a, float3 b)
    { return make_float3(a.x/b.x,a.y/b.y,a.z/b.z); }
    inline __device__ float3 operator*(float3 a, float3 b)
    { return make_float3(a.x*b.x,a.y*b.y,a.z*b.z); }
    inline __device__ float3 operator-(float3 a, float3 b)
    { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
    inline __device__ float3 normalize(float3 v)
    { return (1.f/sqrtf(dot(v,v)))*v; }

    inline __device__ float3 half_way(float3 a, float3 b)
    { return normalize(a+b); }
    
    
    inline __device__ bool all_zero(float3 v)
    { return v.x ==  0.f && v.y == 0.f && v.z == 0.f; }

    inline __device__ float luminance(const float3 &c)
    { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

    inline __device__ float lerp(float x, float y, float s) {
      return x * (1.f - s) + y * s;
    }
    
    inline __device__ float3 lerp(float3 x, float3 y, float s) {
      return x * (1.f - s) + y * s;
    }
    
    inline __device__ float3 reflect(const float3 &i, const float3 &n) {
      return i - 2.f * n * dot(i, n);
    }
    
    inline __device__ float3 refract( float3 i, float3 n, float eta )
    {
      if (eta == 1.f) return i;
      if (eta <= 0.f) return make_float3(0.f);
      if (isnan(eta)) return make_float3(0.f);
      if (isinf(eta)) return make_float3(0.f);
      float cosi = dot(-i, n);
      float cost2 = 1.0f - eta * eta * (1.0f - cosi*cosi);
      float3 t = eta*i + ((eta*cosi - sqrt(abs(cost2))) * n);
      return t * ((cost2 > 0.f) ? make_float3(1.f) : make_float3(0.f));
    }

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ==================================================================
    
    struct DisneyMaterial {
      float3 base_color;
      float3 subsurface_color;
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

    inline __device__ bool same_hemisphere(const float3 &w_o, const float3 &w_i, const float3 &n) {
      return dot(w_o, n) * dot(w_i, n) > 0.f;
    }

    inline __device__ bool relative_ior(const float3 &w_o, const float3 &n, float ior, float &eta_o, float &eta_i)
    {
      bool entering = dot(w_o, n) > 0.f;
      eta_i = entering ? 1.f : ior;
      eta_o = entering ? ior : 1.f;
      return entering;
    }

    // Sample the hemisphere using a cosine weighted distribution,
    // returns a vector in a hemisphere oriented about (0, 0, 1)
    inline __device__ float3 cos_sample_hemisphere(float2 u) {
      float2 s = 2.f * u - make_float2(1.f);
      float2 d;
      float radius = 0.f;
      float theta = 0.f;
      if (s.x == 0.f && s.y == 0.f) {
        d = s;
      } else {
        if (fabs(s.x) > fabs(s.y)) {
          radius = s.x;
          theta  = M_PI / 4.f * (s.y / s.x);
        } else {
          radius = s.y;
          theta  = M_PI / 2.f - M_PI / 4.f * (s.x / s.y);
        }
      }
      d = radius * make_float2(cos(theta), sin(theta));
      return make_float3(d.x, d.y, sqrt(max(0.f, 1.f - d.x * d.x - d.y * d.y)));
    }

    inline 
    __device__ float3 spherical_dir(float sin_theta, float cos_theta, float phi) {
      return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    }

    inline 
    __device__ float power_heuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
      float f = n_f * pdf_f;
      float g = n_g * pdf_g;
      return (f * f) / (f * f + g * g);
    }

    inline 
    __device__ float schlick_weight(float cos_theta) {
      return pow(saturate(1.f - cos_theta), 5.f);
    }

    // Complete Fresnel Dielectric computation, for transmission at ior near 1
    // they mention having issues with the Schlick approximation.
    // eta_i: material on incident side's ior
    // eta_t: material on transmitted side's ior
    inline 
    __device__ float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
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
    __device__ float gtr_1(float cos_theta_h, float alpha) {
      if (alpha >= 1.f) {
        return M_1_PI;
      }
      float alpha_sqr = alpha * alpha;
      return M_1_PI * (alpha_sqr - 1.f) / (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
    }

    // D_GTR2: Generalized Trowbridge-Reitz with gamma=2
    // Burley notes eq. 8
    inline 
    __device__ float gtr_2(float cos_theta_h, float alpha) {
      float alpha_sqr = alpha * alpha;
      return M_1_PI * alpha_sqr / max(pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h), SMALL_EPSILON);
    }

    // D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
    // Burley notes eq. 13
    inline 
    __device__ float gtr_2_aniso(float h_dot_n, float h_dot_x, float h_dot_y, float2 alpha) {
      return M_1_PI / max((alpha.x * alpha.y * pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n)), SMALL_EPSILON);
    }

    inline 
    __device__ float smith_shadowing_ggx(float n_dot_o, float alpha_g) {
      float a = alpha_g * alpha_g;
      float b = n_dot_o * n_dot_o;
      return 1.f / (n_dot_o + sqrt(a + b - a * b));
    }

    inline 
    __device__ float smith_shadowing_ggx_aniso(float n_dot_o, float o_dot_x, float o_dot_y, float2 alpha) {
      return 1.f / (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
    }

    inline 
    // Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using the random samples in s
    __device__ float3 sample_lambertian_dir(const float3 &n, const float3 &v_x, const float3 &v_y, const float2 &s) {
      const float3 hemi_dir = normalize(cos_sample_hemisphere(s));
      return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
    }

    inline 
    // Sample the microfacet normal vectors for the various microfacet distributions
    __device__ float3 sample_gtr_1_h(const float3 &n, const float3 &v_x, const float3 &v_y, float alpha, const float2 &s) {
      float phi_h = 2.f * M_PI * s.x;
      float alpha_sqr = alpha * alpha;
      float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
      float cos_theta_h = sqrt(cos_theta_h_sqr);
      float sin_theta_h = 1.f - cos_theta_h_sqr;
      float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
      return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
    }

    inline 
    __device__ float3 sample_gtr_2_h(const float3 &n, const float3 &v_x, const float3 &v_y, float alpha, const float2 &s) {
      float phi_h = 2.f * M_PI * s.x;
      float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
      float cos_theta_h = sqrt(cos_theta_h_sqr);
      float sin_theta_h = 1.f - cos_theta_h_sqr;
      float3 hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
      return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
    }

    inline 
    __device__ float3 sample_gtr_2_aniso_h(const float3 &n, const float3 &v_x, const float3 &v_y, const float2 &alpha, const float2 &s) {
      float x = 2.f * M_PI * s.x;
      float3 w_h = sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
      return normalize(w_h);
    }

    inline 
    __device__ float lambertian_pdf(const float3 &w_i, const float3 &n) {
      float d = dot(w_i, n);
      if (d > 0.f) {
        return d * M_1_PI;
      }
      return 0.f;
    }

    inline 
    __device__ float gtr_1_pdf(const float3 &w_o, const float3 &w_i, const float3 &w_h, const float3 &n, float alpha) {
      if (!same_hemisphere(w_o, w_i, n)) {
        return 0.f;
      }
      float cos_theta_h = dot(n, w_h);
      float d = gtr_1(cos_theta_h, alpha);
      return d * cos_theta_h / (4.f * dot(w_o, w_h));
    }

    inline 
    __device__ float gtr_2_pdf(const float3 &w_o, const float3 &w_i, const float3 &w_h, const float3 &n, float alpha) {
      if (!same_hemisphere(w_o, w_i, n)) {
        return 0.f;
      }
      float cos_theta_h = fabs(dot(n, w_h));
      float d = gtr_2(cos_theta_h, alpha);
      return d * cos_theta_h / (4.f * fabs(dot(w_o, w_h)));
    }

    inline 
    __device__ float gtr_2_transmission_pdf(const float3 &w_o, const float3 &w_i, const float3 &n, float transmission_roughness, float ior)
    {
      float alpha = max(0.001f, transmission_roughness * transmission_roughness);

      if (same_hemisphere(w_o, w_i, n)) {
        return 0.f;
      }

      // return mat.base_color * abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;


      float eta_o, eta_i;
      bool entering = relative_ior(w_o, n, ior, eta_o, eta_i);
	
      // From Eq 16 of Microfacet models for refraction
      float3 w_ht = -(w_o * eta_i + w_i * eta_o);
      w_ht = normalize(w_ht);

      float cos_theta_h = fabs(dot(n, w_ht));
      float d = gtr_2(cos_theta_h, alpha);

      return d;// /** cos_theta_h*/ / (4.f * fabs(dot(w_o, w_ht)));

      // float3 w_r = refract(-w_o, (entering) ? w_ht : -w_ht, eta_o / eta_i);

      // // float3 n_ = (dot(w_ht, n) > 0) ? n : -;

      // float d = D(w_ht, n, alpha); 
      // float f = F(w_i, w_ht, eta_o, eta_i);
      // float g = G(w_i, w_o, w_ht, alpha); 

      // float i_dot_h = fabs(dot(w_i, w_ht));
      // float o_dot_h = fabs(dot(w_o, w_ht));
      // float i_dot_n = fabs(dot(w_i, n));
      // float o_dot_n = fabs(dot(w_o, n));

      // if (o_dot_n == 0.f || i_dot_n == 0.f) {
      // 	return make_float3(0.f);
      // }

      // // From Eq 21 of Microfacet models for refraction
      // float c = (fabs(i_dot_h) * fabs(o_dot_h)) / (fabs(i_dot_n) * fabs(o_dot_n));
      // c *= pow2(eta_o) / pow2(eta_i * i_dot_h + eta_o * o_dot_h);

      // //// hacking in a spherical gaussian here... Can't seem to get microfacet model working...
      // return mat.base_color * abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;

      // return 1.f;
    }

    // __device__ float gtr_2_transmission_pdf(const float3 &w_o, const float3 &w_i, const float3 &n,
    // 	float alpha, float ior)
    // {
    // 	if (same_hemisphere(w_o, w_i, n)) {
    // 		return 0.f;
    // 	}
    // 	float eta_o, eta_i;
    // 	relative_ior(w_o, n, ior, eta_i, eta_o);

    // 	// From Eq 16 of Microfacet models for refraction
    // 	float3 w_h = -(w_o * eta_o + w_i * eta_i);
    // 	w_h = normalize(w_h);

    // 	// // float3 w_h = normalize(w_i + w_o);
    // 	// float cos_theta_h = fabs(dot(n, w_h));
    // 	// float d = gtr_2(cos_theta_h, alpha);
    // 	// return d * cos_theta_h / (4.f * fabs(dot(w_o, w_h)));


    // 	// float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
    // 	float cos_theta_h = fabs(dot(n, w_h));
    // 	float i_dot_h = dot(w_i, w_h);
    // 	float o_dot_h = dot(w_o, w_h);
    // 	float d = gtr_2(cos_theta_h, alpha);
    // 	float dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
    // 	return d * cos_theta_h * fabs(dwh_dwi);
    // }

    inline 
    __device__ float gtr_2_aniso_pdf(const float3 &w_o, const float3 &w_i, const float3 &w_h, const float3 &n,
                                     const float3 &v_x, const float3 &v_y, const float2 alpha)
    {
      if (!same_hemisphere(w_o, w_i, n)) {
        return 0.f;
      }
      float cos_theta_h = dot(n, w_h);
      float d = gtr_2_aniso(cos_theta_h, fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
      return d * cos_theta_h / (4.f * dot(w_o, w_h));
    }

    inline 
    __device__ float3 disney_diffuse_color(const DisneyMaterial &mat, const float3 &n,
                                           const float3 &w_o, const float3 &w_i, const float3 &w_h)
    {
      return mat.base_color;
    }

    inline 
    __device__ float3 disney_subsurface_color(const DisneyMaterial &mat, const float3 &n,
                                              const float3 &w_o, const float3 &w_i)
    {
      return mat.subsurface_color;
    }

    inline 
    __device__ void disney_diffuse(const DisneyMaterial &mat, const float3 &n,
                                   const float3 &w_o, const float3 &w_i, const float3 &w_h, float3 &bsdf, float3 &color)
    {
      float n_dot_o = fabs(dot(w_o, n));
      float n_dot_i = fabs(dot(w_i, n));
      float i_dot_h = dot(w_i, w_h);
      float fd90 = 0.5f + 1.0f * mat.roughness * i_dot_h * i_dot_h;
      float fi = schlick_weight(n_dot_i);
      float fo = schlick_weight(n_dot_o);
      color = disney_diffuse_color(mat, n, w_o, w_i, w_h);
      bsdf = make_float3(M_1_PI * lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo));
    }

    inline 
    __device__ void disney_subsurface(const DisneyMaterial &mat, const float3 &n,
                                      const float3 &w_o, const float3 &w_i, const float3 &w_h, float3 &bsdf, float3 &color) {
      float n_dot_o = fabs(dot(w_o, n));
      float n_dot_i = fabs(dot(w_i, n));
      float i_dot_h = dot(w_i, w_h);

      float FL = schlick_weight(n_dot_i), FV = schlick_weight(n_dot_o);
      float Fss90 = i_dot_h*i_dot_h * mat.roughness;
      float Fss = lerp(1.f, Fss90, FL) * lerp(1.f, Fss90, FV);
      float ss = 1.25f * (Fss * (1. / (n_dot_i + n_dot_o) - .5f) + .5f);
      color = disney_subsurface_color(mat, n, w_o, w_i);
      bsdf = make_float3(M_1_PI * ss);
    }

    inline 
    // Eavg in the algorithm is fitted into this
    __device__ float AverageEnergy(float rough){
      float smoothness = 1.0f - rough;
      float r = -0.0761947f - 0.383026f * smoothness;
      r = 1.04997f + smoothness * r;
      r = 0.409255f + smoothness * r;
      return min(0.9f, r); 
    }

    inline 
    // multiple scattering...
    // Favg in the algorithm is fitted into this
    __device__ float3 AverageFresnel(float3 specularColor){
      return specularColor + (make_float3(1.f) - specularColor) * (1.f / 21.f);
    }

    inline
    __device__ float3 disney_multiscatter(const DisneyMaterial &mat, const float3 &n,
                                          const float3 &w_o, const float3 &w_i, const float3 &w_h,
                                          cudaTextureObject_t GGX_E_LOOKUP, cudaTextureObject_t GGX_E_AVG_LOOKUP)
    {
#ifdef __CUDA_ARCH__
      float v_dot_n = abs(dot(w_o, n));
      float l_dot_n = abs(dot(w_i, n));
      float alpha = max(mat.roughness*mat.roughness, MIN_ALPHA);

      //E(μ) is in fact the sum of the red and green channels in our environment BRDF
      // vec2 sampleE_o = texture2D(BRDFlut, vec2(NdotV, alpha)).xy;
      // E_o = sampleE_o.x + sampleE_o.y;
      float E_o = tex2D<float>(GGX_E_LOOKUP, v_dot_n, alpha);
      float oneMinusE_o = 1.0 - E_o;
      float E_i = tex2D<float>(GGX_E_LOOKUP, l_dot_n, alpha);
      float oneMinusE_i = 1.0 - E_i;

      float Eavg = tex2D<float>(GGX_E_AVG_LOOKUP, alpha, .5);
      // float Eavg = AverageEnergy(alpha);
      float oneMinusEavg = 1.0 - Eavg;
	
      float lum = luminance(mat.base_color);
      float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
      float3 F0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);
      float3 Favg = AverageFresnel(F0);

      float brdf = (oneMinusE_o * oneMinusE_i) / (M_PI * oneMinusEavg);
      float3 energyScale = (Favg * Favg * Eavg) / (make_float3(1.0) - Favg * oneMinusEavg);

      return brdf * energyScale;
#else
      return make_float3(0.f);
#endif
    }

    // __device__ float G(float3 i, float3 o, float3 h, float alpha)
    // {
    // 	alpha = 1.f - alpha;
    // 	// Roughly follows Eq 23 from Microfacet Models for Refraction.
    // 	// G is approximately the seperable product of two monodirectional shadowing terms G1 (aka smith shadowing function)
    // 	return smith_shadowing_ggx(fabs(dot(i, h)), alpha) * smith_shadowing_ggx(fabs(dot(o, h)), alpha);
    // }

    // __device__ float D(float3 m, float3 n, float alpha)
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

    // __device__ float F(float3 i, float3 m, float eta_t, float eta_i)
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

    inline 
    __device__ float3 disney_microfacet_reflection_color(const DisneyMaterial &mat, const float3 &n,
                                                         const float3 &w_o, const float3 &w_i, const float3 &w_h)
    {
      float lum = luminance(mat.base_color);
      float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
      float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

      float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
      float3 f = lerp(spec, make_float3(1.f), schlick_weight(dot(w_o, n)));
      // float3 f = lerp(spec, make_float3(1.f), schlick_weight(dot(w_i, w_h)));
      return f;
    }

    inline 
    __device__ float3 disney_microfacet_isotropic(const DisneyMaterial &mat, const float3 &n,
                                                  const float3 &w_o, const float3 &w_i, const float3 &w_h)
    {
      float lum = luminance(mat.base_color);
      float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
      float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

      float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
      float d = gtr_2(fabs(dot(n, w_h)), alpha);
      // Finding dot(w_o, n) to be less noisy, but doesn't look as good for crazy normal maps compared to dot(w_i, w_h)
      // Also finding fresnel to be introducing unwanted energy for smooth plastics, so I'm adding a correction term.
      float3 f = lerp(spec, make_float3(1.f), schlick_weight(fabs(dot(w_i, w_h))) * lerp(.5f, 1.f, max(mat.metallic, alpha)));
      float g = smith_shadowing_ggx(fabs(dot(n, w_i)), alpha) * smith_shadowing_ggx(fabs(dot(n, w_o)), alpha);
      return d * f * g;
    }

    inline 
    __device__ float3 disney_microfacet_transmission_color(const DisneyMaterial &mat, const float3 &n,
                                                           const float3 &w_o, const float3 &w_i, const float3 &w_h)
    {	
      // Approximate absorption
      // note that compositing will be incorrect...
      return mat.base_color;
    }

    inline 
    __device__ void disney_microfacet_transmission_isotropic(const DisneyMaterial &mat, const float3 &n,
                                                             const float3 &w_o, const float3 &w_i, float &bsdf, float3 &color)
    {	

      float eta_o, eta_i;
      bool entering = relative_ior(w_o, n, mat.ior, eta_o, eta_i);

      float alpha = max(0.001f, mat.transmission_roughness * mat.transmission_roughness);
	
      // From Eq 16 of Microfacet models for refraction
      float3 w_ht = -(w_o * eta_i + w_i * eta_o);
      w_ht = normalize(w_ht);

      color = disney_microfacet_transmission_color(mat, n, w_o, w_i, w_ht);

      float lum = luminance(color);
      float3 tint = lum > 0.f ? color / lum : make_float3(1.f);
	
      float3 spec = disney_microfacet_transmission_color(mat, n, w_o, w_i, w_ht);

      // float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
      float cos_theta_h = fabs(dot(n, w_ht));
      float d = gtr_2(cos_theta_h, alpha);
      float3 f = lerp(spec, make_float3(1.f), 1.0f - schlick_weight(dot(w_i, w_ht)));
      float g = smith_shadowing_ggx(abs(dot(n, w_i)), alpha) * smith_shadowing_ggx(abs(dot(n, w_o)), alpha);
	
      bsdf = d;
      color = spec;
      return;// * f;// * g; // * f * g;



      // // w_ht = n; // HACK

      // float3 w_r = refract(-w_o, (entering) ? w_ht : -w_ht, eta_o / eta_i);

      // // float3 n_ = (dot(w_ht, n) > 0) ? n : -;

	

      // // float d = gtr_2();//D(w_ht, n, alpha); 
      // float f = F(w_i, w_ht, eta_o, eta_i);
      // float g = G(w_i, w_o, w_ht, alpha); 

      // float i_dot_h = fabs(dot(w_i, w_ht));
      // float o_dot_h = fabs(dot(w_o, w_ht));
      // float i_dot_n = fabs(dot(w_i, n));
      // float o_dot_n = fabs(dot(w_o, n));

      // if (o_dot_n == 0.f || i_dot_n == 0.f) {
      // 	return make_float3(0.f);
      // }

      // // From Eq 21 of Microfacet models for refraction
      // float c = (fabs(i_dot_h) * fabs(o_dot_h)) / (fabs(i_dot_n) * fabs(o_dot_n));
      // c *= pow2(eta_o) / pow2(eta_i * i_dot_h + eta_o * o_dot_h);

      // //// hacking in a spherical gaussian here... Can't seem to get microfacet model working...
      // return mat.base_color * d;// * f * g; //abs( pow(dot(w_ht, n), (1.0f / (alpha + EPSILON))) ); //* c;//g; //c * (1.f - f) * g * d;
    }

    inline 
    __device__ float3 disney_microfacet_anisotropic(const DisneyMaterial &mat, const float3 &n,
                                                    const float3 &w_o, const float3 &w_i, const float3 &w_h, const float3 &v_x, const float3 &v_y)
    {
      float lum = luminance(mat.base_color);
      float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
      float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint), mat.base_color, mat.metallic);

      float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
      float a = mat.roughness * mat.roughness;
      float2 alpha = make_float2(max(MIN_ALPHA, a / aspect), max(MIN_ALPHA, a * aspect));
      float d = gtr_2_aniso(fabs(dot(n, w_h)), fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
      // Finding dot(w_o, n) to be less noisy, but doesn't look as good for crazy normal maps compared to dot(w_i, w_h)
      // Also finding fresnel to be introducing unwanted energy for smooth plastics, so I'm adding a correction term.
      float3 f = lerp(spec, make_float3(1.f), schlick_weight(fabs(dot(w_i, w_h))) * lerp(.5f, 1.f, max(mat.metallic, alpha.x * alpha.y)));
      float g = smith_shadowing_ggx_aniso(fabs(dot(n, w_i)), fabs(dot(w_i, v_x)), fabs(dot(w_i, v_y)), alpha)
        * smith_shadowing_ggx_aniso(fabs(dot(n, w_o)), fabs(dot(w_o, v_x)), fabs(dot(w_o, v_y)), alpha);
      return d * f * g;
    }

    inline 
    __device__ float disney_clear_coat(const DisneyMaterial &mat, const float3 &n,
                                       const float3 &w_o, const float3 &w_i, const float3 &w_h)
    {
      float alpha = lerp(0.1f, MIN_ALPHA, mat.clearcoat_gloss);
      float d = gtr_1(fabs(dot(n, w_h)), alpha);
      float f = lerp(0.04f, 1.f, schlick_weight(dot(w_i, n)));
      float g = smith_shadowing_ggx(fabs(dot(n, w_i)), 0.25f) * smith_shadowing_ggx(fabs(dot(n, w_o)), 0.25f);
      return /*0.25f * */mat.clearcoat * d * f * g;
    }

    inline 
    __device__ float3 disney_sheen(const DisneyMaterial &mat, const float3 &n,
                                   const float3 &w_o, const float3 &w_i, const float3 &w_h)
    {
      float lum = luminance(mat.base_color);
      float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
      float3 sheen_color = lerp(make_float3(1.f), tint, mat.sheen_tint);
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
    __device__ void disney_brdf(
                                const DisneyMaterial &mat, 
                                const float3 &g_n,
                                const float3 &s_n,
                                const float3 &b_n,
                                const float3 &v_x, 
                                const float3 &v_y,
                                const float3 &w_o, 
                                const float3 &w_i, 
                                const float3 &w_h, 
                                float3 &bsdf
                                ) {
      // initialize bsdf value to black for now.
      bsdf = make_float3(0.f);

      // transmissive objects refract when back of surface is visible.
      if (!same_hemisphere(w_o, w_i, b_n) && (mat.specular_transmission > 0.f)) {
        float spec_trans; float3 trans_color;
        disney_microfacet_transmission_isotropic(mat, b_n, w_o, w_i, spec_trans, trans_color);
        spec_trans = spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
        bsdf = make_float3(spec_trans) * trans_color;			
        return;
      }

      float coat = disney_clear_coat(mat, b_n, w_o, w_i, w_h);
      float3 sheen = disney_sheen(mat, b_n, w_o, w_i, w_h);
      float3 diffuse_bsdf, diffuse_color;
      disney_diffuse(mat, b_n, w_o, w_i, w_h, diffuse_bsdf, diffuse_color);
      float3 subsurface_bsdf, subsurface_color;
      disney_subsurface(mat, b_n, w_o, w_i, w_h, subsurface_bsdf, subsurface_color);
      float3 gloss;
      if (mat.anisotropy == 0.f) {
        gloss = disney_microfacet_isotropic(mat, b_n, w_o, w_i, w_h);
        // gloss = gloss + disney_multiscatter(mat, n, w_o, w_i, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
      } else 
	{
          gloss = disney_microfacet_anisotropic(mat, b_n, w_o, w_i, w_h, v_x, v_y);
          // gloss = gloss + disney_multiscatter(mat, n, w_o, w_i, GGX_E_LOOKUP, GGX_E_AVG_LOOKUP);
	}
	
      bsdf = (lerp(diffuse_bsdf * diffuse_color, 
                   subsurface_bsdf * subsurface_color, 
                   mat.flatness) 
              * (1.f - mat.metallic) * (1.f - mat.specular_transmission) 
              + sheen + coat + gloss) * fabs(dot(w_i, b_n));
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
    __device__ void disney_pdf(
                               const DisneyMaterial &mat, 
                               const float3 &g_n,
                               const float3 &s_n,
                               const float3 &b_n,
                               const float3 &v_x, 
                               const float3 &v_y,
                               const float3 &w_o, 
                               const float3 &w_i, 
                               const float3 &w_h, 
                               float &pdf
                               ) {
      pdf = 0.f;

      bool entering = dot(w_o, b_n) > 0.f;
      bool sameHemisphere = same_hemisphere(w_o, w_i, b_n);
	
      float alpha = max(0.002f, mat.roughness * mat.roughness);
      float t_alpha = max(0.002f, mat.transmission_roughness * mat.transmission_roughness);
      float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
      float2 alpha_aniso = make_float2(max(0.002f, alpha / aspect), max(0.002f, alpha * aspect));

      float clearcoat_alpha = lerp(0.1f, MIN_ALPHA, mat.clearcoat_gloss);

      float diffuse = lambertian_pdf(w_i, b_n);
      float clear_coat = gtr_1_pdf(w_o, w_i, w_h, b_n, clearcoat_alpha);

      float n_comp = 3.f;
      float microfacet = 0.f;
      float microfacet_transmission = 0.f;
      if (mat.anisotropy == 0.f) {
        microfacet = gtr_2_pdf(w_o, w_i, w_h, b_n, alpha);
      } else {
        microfacet = gtr_2_aniso_pdf(w_o, w_i, w_h, b_n, v_x, v_y, alpha_aniso);
      }

      if ((mat.specular_transmission > 0.f) && (!same_hemisphere(w_o, w_i, b_n))) {
        microfacet_transmission = gtr_2_transmission_pdf(w_o, w_i, b_n, mat.transmission_roughness, mat.ior);
      } 

      // not sure why, but energy seems to be added from smooth metallic. By subtracting mat.metallic from n_comps,
      // we decrease brightness and become almost perfectly conserving energy for shiny metallic. As metals get 
      // rough, we lose energy from our single scattering microfacet model around .1 roughness, so we 
      // remove the energy reduction kludge for metallic in the case of slightly rough metal. We still 
      // seem to lose a lot of energy in that case, and could likely benefit from a multiple scattering microfacet
      // model. 
      // For transmission, so long as we subtract 1 from the components, we seem to preserve energy
      // regardless if the transmission is rough or smooth.
      float metallic_kludge = mat.metallic;
      float transmission_kludge = mat.specular_transmission;
      n_comp -= lerp(transmission_kludge, metallic_kludge, mat.metallic); 
      pdf = (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
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
    __device__ void sample_disney_brdf(
                                       const DisneyMaterial &mat,
                                       LCGRand &rng,
                                       const float3 &g_n, const float3 &s_n, const float3 &b_n, 
                                       const float3 &v_x, const float3 &v_y,
                                       const float3 &w_o,
                                       float3 &w_i, 
                                       float &pdf, 
                                       int &sampled_bsdf, 
                                       float3 &bsdf
                                       ) {
      // Randomly pick a brdf to sample
      if (mat.specular_transmission == 0.f) {
        sampled_bsdf = lcg_randomf(rng) * 3.f;
        sampled_bsdf = clamp(sampled_bsdf, 0, 2);
      } else {
        // If we're looking at the front face 
        if (dot(w_o, b_n) > 0.f) {
          sampled_bsdf = lcg_randomf(rng) * 4.f;
          sampled_bsdf = clamp(sampled_bsdf, 0, 3);
        }
        else sampled_bsdf = DISNEY_TRANSMISSION_BRDF; 
      }

      float2 samples = make_float2(lcg_randomf(rng), lcg_randomf(rng));
      if (sampled_bsdf == DISNEY_DIFFUSE_BRDF) {
        w_i = sample_lambertian_dir(b_n, v_x, v_y, samples);
      } else if (sampled_bsdf == DISNEY_GLOSSY_BRDF) {
        float3 w_h;
        float alpha = max(MIN_ALPHA, mat.roughness * mat.roughness);
        if (mat.anisotropy == 0.f) {
          w_h = sample_gtr_2_h(b_n, v_x, v_y, alpha, samples);
        } else {
          float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
          float2 alpha_aniso = make_float2(max(MIN_ALPHA, alpha / aspect), max(MIN_ALPHA, alpha * aspect));
          w_h = sample_gtr_2_aniso_h(b_n, v_x, v_y, alpha_aniso, samples);
        }
        w_i = reflect(-w_o, w_h);

        // Invalid reflection, terminate ray
        if (!same_hemisphere(w_o, w_i, b_n)) {
          pdf = 0.f;
          w_i = make_float3(0.f);
          bsdf = make_float3(0.f);
          return;
        }
      } else if (sampled_bsdf == DISNEY_CLEARCOAT_BRDF) {
        float alpha = lerp(0.1f, MIN_ALPHA, mat.clearcoat_gloss);
        float3 w_h = sample_gtr_1_h(b_n, v_x, v_y, alpha, samples);
        w_i = reflect(-w_o, w_h);

        // Invalid reflection, terminate ray
        if (!same_hemisphere(w_o, w_i, b_n)) {
          pdf = 0.f;
          w_i = make_float3(0.f);
          bsdf = make_float3(0.f);
          return;
        }
      } else if (sampled_bsdf == DISNEY_TRANSMISSION_BRDF) {	
        float alpha = max(MIN_ALPHA, mat.transmission_roughness * mat.transmission_roughness);
        float3 w_h = sample_gtr_2_h(b_n, v_x, v_y, alpha, samples);
        float eta_o, eta_i;
        bool entering = relative_ior(w_o, w_h, mat.ior, eta_o, eta_i);
        // w_i = refract(-w_o, w_h, eta_o / eta_i);

        // w_o is flipped, so we also flip eta
        w_i = refract(-w_o, (entering) ? w_h : -w_h, eta_i / eta_o);

        // Total internal reflection
        if (all_zero(w_i)) {
          w_i = reflect(-w_o, (entering) ? w_h : -w_h);
          pdf = 1.f;
          bsdf = make_float3(1.f);// Normally absorption would happen here...
          return;
        }
      }
	
      float3 w_h = normalize(w_i + w_o);
      disney_pdf(mat, g_n, s_n, b_n, v_x, v_y, w_o, w_i, w_h, pdf);
      disney_brdf(mat, g_n, s_n, b_n, v_x, v_y, w_o, w_i, w_h, bsdf);
    }

    
  }
  
  namespace render {

    struct Blender {
      struct HitBSDF : public nvisii::DisneyMaterial {
        inline __device__
        vec3f getAlbedo(bool dbg=false) const
        { return Lambert((const vec3f&)base_color).getAlbedo(dbg); }
        
        inline __device__
        EvalRes eval(render::DG dg, vec3f _wi, bool dbg=false) const
        {
#if 1
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
          float3 g_n = (const float3 &)dg.Ng;
          float3 s_n = (const float3 &)dg.Ns;
          float3 b_n = (const float3 &)dg.Ns;
          linear3f localFrame = owl::common::frame(dg.Ns);
          float3 v_x = (const float3 &)localFrame.vx;
          float3 v_y = (const float3 &)localFrame.vy;
          float3 w_o = (const float3 &)dg.wo;
          float3 w_i = (const float3 &)_wi;
          float3 w_h = nvisii::half_way(w_o,w_i);
          float3 bsdf;
          disney_brdf(*this,g_n,s_n,b_n,v_x,v_y,w_o,w_i,w_h,bsdf);
          EvalRes res;
          (float3&)res.value = bsdf;
          res.pdf = 1.f;
          return res;
#else
          return Lambert((const vec3f&)base_color).eval(dg,_wi,dbg);
#endif
        }

        inline __device__
        SampleRes sample(const DG &dg,
                         Random &rng,
                         bool dbg = false)
        {
          linear3f localFrame = owl::common::frame(dg.Ns);
          // vec3f N = dg.Ng;
          
          // * Sample a component of the Disney BRDF
          // * @param mat The structure containing material information.
          // * @param rng The random number generator
          // * @param g_n The geometric normal (cross product of the two triangle edges)
          // * @param s_n The shading normal (per-vertex interpolated normal)
          // * @param b_n The bent normal (see A.3 here https://arxiv.org/abs/1705.01263)
          // * @param v_x The tangent vector
          // * @param v_y The binormal vector
          // * @param w_o The outgoing (aka view) vector
          // * @param w_i The returned incoming (aka light) vector
          // * @param pdf The probability of this sample, for importance sampling
          // * @param sampled_bsdf Enum for which bsdf was sampled. 
          // * 	Can be either DISNEY_DIFFUSE_BRDF, DISNEY_GLOSSY_BRDF, DISNEY_CLEARCOAT_BRDF, DISNEY_TRANSMISSION_BRDF
          // * @param bsdf The throughput of all brdfs in the sampled direction
          // */
          // device__ void sample_disney_brdf(
          // 	const DisneyMaterial &mat,
          // 	LCGRand &rng,
          // 	const float3 &g_n, const float3 &s_n, const float3 &b_n, 
          // 	const float3 &v_x, const float3 &v_y,
          // 	const float3 &w_o,
          // 	float3 &w_i, 
          // 	float &pdf, 
          // 	int &sampled_bsdf, 
          // 	float3 &bsdf
          // )
          
          // in:
          float3 g_n;
          float3 s_n;
          float3 b_n;
          float3 v_x;
          float3 v_y;
          float3 w_o;
          // out:
          float3 w_i;
          float pdf;
          int sampled_bsdf;
          float3 bsdf;

          (vec3f&)g_n = dg.Ng;
          (vec3f&)s_n = dg.Ns;
          (vec3f&)b_n = dg.Ns;
          (vec3f&)v_x = localFrame.vx;
          (vec3f&)v_y = localFrame.vy;
          (vec3f&)w_o = dg.wo;
          // sample_disney_brdf(*this, rng, v_gz, v_z, v_bz, v_x, v_y, w_o, // inputs
          //                    w_i, bsdfPDF, sampledBsdf, bsdf);           // outputs
          sample_disney_brdf(*this, rng, g_n,s_n,b_n,v_x,v_y,w_o,
                             // out:
                             w_i, pdf, sampled_bsdf, bsdf);           // outputs
          
          SampleRes res;
          (float3&)res.wi = w_i;
          res.pdf = pdf;
          (float3&)res.weight = bsdf;
          switch(sampled_bsdf) {
          case DISNEY_DIFFUSE_BRDF:
            res.type = BSDF_DIFFUSE_REFLECTION;
            break;
          case DISNEY_GLOSSY_BRDF:
            res.type = BSDF_GLOSSY_REFLECTION;
            break;
          case DISNEY_CLEARCOAT_BRDF:
            res.type = BSDF_SPECULAR_REFLECTION;
            break;
          case DISNEY_TRANSMISSION_BRDF:
            res.type = BSDF_SPECULAR_TRANSMISSION;
            break;
          };
          return res;
        }
        

        enum { bsdfType = Lambert::bsdfType };
        // enum { bsdfType = Minneart::bsdfType | Lambert::bsdfType };
      };
      struct DD {
        inline __device__
        void make(HitBSDF &multi, vec3f geometryColor, bool dbg) const
        {
          // multi.base_color = sampleTexture
          //   (p.base_color_texture_id, uv, make_float3(.8f, .8f, .8f));
          (vec3f&)multi.base_color
            = !isnan(geometryColor.x)
            ? geometryColor
            : (const vec3f&)base_color;

//           #define MIN_ROUGHNESS .04f
// src/nvisii/devicecode/types.h:#define MIN_ALPHA MIN_ROUGHNESS * MIN_ROUGHNESS
          
          // multi.metallic = sampleTexture
          //   (p.metallic_texture_id, uv, p.metallic_texture_channel, .0f);
          multi.metallic = this->metallic;
          // multi.specular = sampleTexture
          //   (p.specular_texture_id, uv, p.specular_texture_channel, .5f);
          multi.specular = this->specular;
          // multi.roughness = sampleTexture
          //   (p.roughness_texture_id, uv, p.roughness_texture_channel, .5f);
          multi.roughness = this->roughness;
          // multi.specular_tint = sampleTexture
          //   (p.specular_tint_texture_id, uv, p.specular_tint_texture_channel, 0.f);
          multi.specular_tint = this->specular_tint;
          // multi.anisotropy = sampleTexture
          //   (p.anisotropic_texture_id, uv, p.anisotropic_texture_channel, 0.f);
          multi.anisotropy = this->anisotropic;
          // multi.sheen = sampleTexture
          //   (p.sheen_texture_id, uv, p.sheen_texture_channel, 0.f);
          multi.sheen = this->sheen;
          // multi.sheen_tint = sampleTexture
          //   (p.sheen_tint_texture_id, uv, p.sheen_tint_texture_channel, 0.5f);
          multi.sheen_tint = this->sheen_tint;
          // multi.clearcoat = sampleTexture
          //   (p.clearcoat_texture_id, uv, p.clearcoat_texture_channel, 0.f);
          multi.clearcoat = this->clearcoat;
          // float clearcoat_roughness = sampleTexture
          //   (p.clearcoat_roughness_texture_id, uv, p.clearcoat_roughness_texture_channel, 0.3f);
          float clearcoat_roughness = this->clearcoat_roughness;
          // multi.ior = sampleTexture
          //   (p.ior_texture_id, uv, p.ior_texture_channel, 1.45f);
          multi.ior = this->ior;
          // multi.specular_transmission = sampleTexture
          //   (p.transmission_texture_id, uv, p.transmission_texture_channel, 0.f);
          multi.specular_transmission = this->transmission;
          // multi.flatness = sampleTexture
          //   (p.subsurface_texture_id, uv, p.subsurface_texture_channel, 0.f);
          multi.flatness = this->subsurface;
          // multi.subsurface_color = sampleTexture
          //   (p.subsurface_color_texture_id, uv, make_float3(0.8f, 0.8f, 0.8f));
          (vec3f&)multi.subsurface_color = this->subsurface_color;
          // multi.transmission_roughness = sampleTexture
          //   (p.transmission_roughness_texture_id, uv, p.transmission_roughness_texture_channel, 0.f);
          multi.transmission_roughness = this->transmission_roughness;
          // multi.alpha = sampleTexture
          //   (p.alpha_texture_id, uv, p.alpha_texture_channel, 1.f);
          multi.alpha = this->alpha;
          
          // multi.transmission_roughness
          //   = max(max(multi.transmission_roughness, MIN_ROUGHNESS), roughnessMinimum);
          multi.transmission_roughness
            = max(multi.transmission_roughness, MIN_ROUGHNESS);
          // multi.roughness = max(max(multi.transmission_roughness, MIN_ROUGHNESS), roughnessMinimum);
          // iw:
          multi.roughness = MIN_ROUGHNESS+multi.roughness*(1.f-2.f*MIN_ROUGHNESS);
          
          // clearcoat_roughness = max(clearcoat_roughness, MIN_ROUGHNESS);
          clearcoat_roughness = max(clearcoat_roughness, MIN_ROUGHNESS);
          multi.clearcoat_gloss = 1.f - clearcoat_roughness * clearcoat_roughness;
        }
	vec3f base_color;
	vec3f subsurface_color;
	vec3f subsurface_radius;
        float subsurface;
	float metallic;

	float specular;
	float roughness;
	float specular_tint;
	float anisotropic;
	float anisotropic_rotation;

	float sheen;
	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;
	float clearcoat_roughness;

	float ior;
	float transmission;
	float transmission_roughness;
	float flatness;
	float alpha;
        // vec3f baseColor;
      };
    };
    
  }
}
