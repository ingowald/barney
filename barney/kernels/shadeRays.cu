// ======================================================================== //
// Copyright 2023-2025 Ingo Wald                                            //
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

#include "barney/fb/FrameBuffer.h"
#include "barney/fb/TiledFB.h"
#include "barney/render/World.h"
#include "barney/render/Renderer.h"
#include "barney/GlobalModel.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  namespace render {

#define SCI_VIS_MODE 0
    
#define MAX_DIFFUSE_BOUNCES 3
    
#define ENV_LIGHT_SAMPLING 1

#define USE_MIS 1


#define CLAMP_F_R 3.f


#if RTC_DEVICE_CODE
    inline __rtc_device float square(float f) { return f*f; }
  
    
    enum { MAX_PATH_DEPTH = 10 };

    inline __rtc_device
    float safe_eps(float f, vec3f v)
    {
      return max(f,1e-5f*reduce_max(abs(v)));
    }

    
    inline __rtc_device
    vec3f randomDirection(Random &rng)
    {
      vec3f v;
      while (true) {
        v.x = 1.f-2.f*rng();
        v.y = 1.f-2.f*rng();
        v.z = 1.f-2.f*rng();
        if (dot(v,v) <= 1.f)
          return normalize(v);
      }
    }

    inline __rtc_device
    bool sampleAreaLights(Light::Sample &ls,
                          const render::World::DD &world,
                          const vec3f P,
                          const vec3f N,
                          Random &random,
                          bool dbg)
    {
      if (world.numQuadLights == 0) return false;
      static const int RESERVOIR_SIZE = 8;
      int   lID[RESERVOIR_SIZE];
      float u[RESERVOIR_SIZE];
      float v[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      QuadLight::DD light;
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numQuadLights),
                     world.numQuadLights-1);
        weights[i] = 0.f;
        light = world.quadLights[lID[i]];
        u[i] = random();
        v[i] = random();
        float lightArea = light.area;
#ifndef NDEBUG
        if (lightArea < 0.f)
          printf("INVALID NEGATIVE LIGHT AREA on light %i/%i : %f\n",
                 lID[i],world.numQuadLights,lightArea);
#endif
        vec3f LN = light.normal;
        vec3f LP = light.corner + u[i]*light.edge0 + v[i]*light.edge1;
        vec3f lightDir = LP - P;
        float lightDist = length(lightDir);
        if (lightDist < 1e-3f) continue;
      
        lightDir *= 1.f/lightDist;

        float weight = dot(lightDir,N);
        if (weight <= 1e-3f) continue;
        weight *= -dot(lightDir,LN);
        if (weight <= 1e-3f) continue;
#ifndef NDEBUG
        if (lightArea == 0.f || reduce_max(light.emission) == 0)
          printf("invalid light! %f : %f %f %f\n",
                 lightArea,
                 light.emission.x,
                 light.emission.y,
                 light.emission.z);
#endif
        weight *= (1.f/(lightDist*lightDist)) * lightArea * reduce_max(light.emission);
#ifndef NDEBUG
        if (isnan(sumWeights) || weight < 0.f)
          printf("area lights: weight[%i:%i] is nan or negative: dist  %f area %f emission %f %f %f\n",
                 i,lID[i],lightDist,lightArea,
                 light.emission.x,
                 light.emission.y,
                 light.emission.z);
#endif
        sumWeights += weight;
        weights[i] = weight;
      }
#ifndef NDEBUG
      if (isnan(sumWeights))
        printf("area lights: sumWeights is nan!\n");
#endif
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.quadLights[lID[i]];
      vec3f LP = light.corner + u[i]*light.edge0 + v[i]*light.edge1;
      vec3f LD = LP-P;
      ls.direction
        = normalize(LD);
      ls.distance
        = length(LD);
      ls.radiance
        = light.emission
        * (light.area * -dot(light.normal,ls.direction)
           / square(ls.distance));
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numQuadLights));
#ifndef NDEBUG
      if (ls.pdf <= 0.f)
        printf("invalid area light PDF %f from i %i weight %f sum %f\n",
               ls.pdf,i,weights[i],sumWeights);
#endif
      return true;
    }

    inline __rtc_device
    bool sampleDirLights(Light::Sample &ls,
                         const World::DD &world,
                         const Renderer::DD &renderer,
                         const vec3f P,
                         const vec3f N,
                         Random &random,
                         bool dbg)
    {
      if (world.numDirLights == 0) return false;
      static const int RESERVOIR_SIZE = 2;
      int   lID[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      DirLight::DD light;
    
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numDirLights),
                     world.numDirLights-1);
        weights[i] = 0.f;
        light = world.dirLights[lID[i]];
        vec3f light_radiance
          = light.color
          * light.radiance;
        
        vec3f lightDir = -light.direction;
        float weight = dot(lightDir,N);
        if (dbg) printf("light #%i, dir %f %f %f weight %f\n",lID[i],lightDir.x,lightDir.y,lightDir.z,weight);
        if (weight <= 1e-3f) continue;
        weight *= reduce_max(light_radiance);
        if (weight <= 1e-3f) continue;
        weights[i] = weight;
        sumWeights += weight;
      }
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.dirLights[lID[i]];
      ls.direction
        = -light.direction;
      ls.distance
        = BARNEY_INF;
      ls.radiance
        = light.radiance;
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numDirLights));
      return weights[i] != 0.f;
    }

    inline __rtc_device
    bool sampleEnvLight(Light::Sample &ls,
                        const World::DD &world,
                        const Renderer::DD &renderer,
                        const vec3f P,
                        const vec3f N,
                        Random &random,
                        bool dbg)
    {
      /* in barney, the environment is either a explicit hdri map (in
         EnvMapLight); or a uniform brightness of 'renderer.ambientRadiance' */
      if (world.envMapLight.texture)
        ls = world.envMapLight.sample(random,dbg);
      else {
#if 0
        ls.direction = randomDirection(random);
        ls.radiance  = renderer.ambientRadiance;
        if (dot(ls.direction,N) < 0.f) ls.direction = -ls.direction;
        ls.pdf       = ONE_OVER_TWO_PI;
        ls.distance  = BARNEY_INF;
#else
        ls.direction = randomDirection(random);
        ls.radiance  = renderer.ambientRadiance;
        ls.pdf       = ONE_OVER_FOUR_PI;
        ls.distance  = BARNEY_INF;
#endif
      }
      return true;
    }

    inline __rtc_device
    bool sampleLights(Light::Sample &ls,
                      const World::DD &world,
                      const Renderer::DD &renderer,
                      const vec3f P,
                      const vec3f Ng,
                      Random &random,
#if USE_MIS
                      bool &lightNeedsMIS,
                      bool &lightIsDirLight,
#endif
                      bool dbg)
    {
#if USE_MIS
# if 0
      // huh ... not sure this is correct; setting this to true means
      // we'll always compute MIS weights for shadow and bounce ray as
      // if there was only an env-map light; even though we may
      // acutally have sampled a dir-light. that _may_ be true because
      // even if we did sample a dirlight there still _is_ a pdf for
      // the env-map light... but it's a bit iffy.
      lightNeedsMIS = true;
# else
      lightNeedsMIS = false;
# endif
#endif

#if ENV_LIGHT_SAMPLING
      Light::Sample els;
      float elsWeight
        = (sampleEnvLight(els,world,renderer,P,Ng,random,dbg)
           ? (reduce_max(els.radiance)/els.pdf)
           : 0.f);
      if (dbg)
        printf("els rad %f %f %f pdf %f\n",
               els.radiance.x,
               els.radiance.y,
               els.radiance.z,
               els.pdf);
      // = world.envMapLight.sample(random,dbg);
#else
      float elsWeight = 0.f;
#endif

      Light::Sample als;
      float alsWeight
        = (sampleAreaLights(als,world,P,Ng,random,dbg)
           ? (reduce_max(als.radiance)/als.pdf)
           : 0.f);
      Light::Sample dls;
      float dlsWeight
        = (sampleDirLights(dls,world,renderer,P,Ng,random,dbg)
           ? (reduce_max(dls.radiance)/dls.pdf)
           : 0.f);

      if (dbg) printf("sampling lights dls %f els %f\n",
                      dlsWeight,elsWeight);
      
      float sumWeights
        = alsWeight+dlsWeight+elsWeight;
      if (sumWeights == 0.f) return false;

      elsWeight *= 1.f/sumWeights;
      alsWeight *= 1.f/sumWeights;
      dlsWeight *= 1.f/sumWeights;
      
      float r = random();
      if (dbg) printf(" light sample %f in cdf %f %f %f\n",
                      r,alsWeight,elsWeight,dlsWeight);
      if (r <= alsWeight) {
        ls = als;
        ls.pdf *= alsWeight;
#if ENV_LIGHT_SAMPLING
      } else if (r <= alsWeight+elsWeight) {
        ls = els;
        ls.pdf *= elsWeight;
        if (dbg) printf(" ->  picked env light sample\n");
# if USE_MIS
        lightNeedsMIS = true;
# endif
#endif
      } else {
        ls = dls;
        ls.pdf *= dlsWeight;
# if USE_MIS
        lightIsDirLight = true;
# endif
        if (dbg) printf(" ->  picked DIR light sample, dls weight %f pdf %f\n",dlsWeight,ls.pdf);
      }
      if (isnan(ls.pdf) || (ls.pdf <= 0.f)) return false;
      
      return true;
    }




    inline __rtc_device
    float schlick(float cosine,
                  float ref_idx)
    {
      float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
      r0 = r0 * r0;
      return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
    }
  
  

    inline __rtc_device
    bool refract(const vec3f& v,
                 const vec3f& n,
                 float ni_over_nt,
                 vec3f &refracted)
    {
      vec3f uv = normalize(v);
      float dt = dot(uv, n);
      float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
      if (discriminant > 0.f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
      }
      else
        return false;
    }
  
    inline __rtc_device
    vec3f radianceFromEnv(const World::DD &world,
                          const Renderer::DD &renderer,
                          Ray &ray)
    {
      auto &env = world.envMapLight;
      if (env.texture) {
        vec3f d = xfmVector(env.toLocal,normalize(ray.dir));
        float theta = pbrtSphericalTheta(d);
        float phi   = pbrtSphericalPhi(d);
        const float invPi  = 1.f/(float)M_PI;
        const float inv2Pi = 1.f/(2.f* (float)M_PI);
        vec2f uv(phi * inv2Pi, theta * invPi);

        vec4f color = rtc::tex2D<vec4f>(env.texture,uv.x,uv.y);
        float envLightPower = 1.f;
        return envLightPower*vec3f(color.x,color.y,color.z);
      } else {
        return renderer.ambientRadiance;
      }
    }

    /*! if there _is_ a dedicated env-map light specified, this looks
      up the background color from that map; otherwise, it returns
      the 'ray.misscolor' that the primary ray generation has set as
      default color for this ray */
    inline __rtc_device
    vec3f primaryRayMissColor(const World::DD &world,
                              const Renderer::DD &renderer,
                              Ray &ray)
    {
      if (world.envMapLight.texture)
        return radianceFromEnv(world,renderer,ray);
      return
        // primary rays do store a default misscolor in the ray itself
        // - we simply return this if there's no env-map.
        (const vec3f&)ray.missColor;
    }

    /*! ugh - that should all go into material::AnariPhysical .... */
    inline __rtc_device
    void bounce(const World::DD &world,
                const Renderer::DD &renderer,
                vec3f &fragment,
                Ray &path,
                Ray &shadowRay,
                int pathDepth)
    {
      
      const float EPS = 1e-4f;

      const bool  hadNoIntersection  = !path.hadHit();
      const vec3f incomingThroughput = path.throughput;
      
#ifdef NDEBUG
      bool dbg = false;
#else
      bool dbg = path.dbg;
#endif
      
      if (dbg)
        printf("(%i) ------------------------------------------------------------------\n -> incoming %f %f %f dir %f %f %f t %f\n  tp %f %f %f ismiss %i, bsdf %i\n",
               pathDepth,
               path.org.x,
               path.org.y,
               path.org.z,
               (float)path.dir.x,
               (float)path.dir.y,
               (float)path.dir.z,
               path.tMax,
               (float)path.throughput.x,
               (float)path.throughput.y,
               (float)path.throughput.z,
               int(hadNoIntersection),(int)path.bsdfType);
      
      if (path.isShadowRay) {
        // ==================================================================
        // shadow ray = all we have to do is add carried radiance if it
        // reached the light, and discards
        // ==================================================================
                 
        if (hadNoIntersection) {
          // fragment = clamp((vec3f)path.throughput,vec3f(0.f),vec3f(1.f));
          fragment =
# if USE_MIS
            (float)path.misWeight *
#endif
            (vec3f)path.throughput;
          if (dbg)
            printf("_shadow_ ray reaches light: tp %f %f %f misweight %f frag %f %f %f\n",
                   (float)path.throughput.x,
                   (float)path.throughput.y,
                   (float)path.throughput.z,
                   (float)path.misWeight,
                   fragment.x,
                   fragment.y,
                   fragment.z);
          if (dbg) printf("shadow miss, frag %f %f %f\n",
                                    fragment.x,
                                    fragment.y,
                                    fragment.z);
        }

        // this path is done.
        shadowRay.tMax = -1.f;
        path.tMax = -1.f;
        return;
      }

      vec3f Ng = path.getN();
      if (dbg) printf("path.N %f %f %f\n",Ng.x,Ng.y,Ng.z);
      const bool  isVolumeHit        = (Ng == vec3f(0.f));
      if (!isVolumeHit)
        Ng = normalize(Ng);
      const bool  hitWasOnFront      = dot((vec3f)path.dir,Ng) < 0.f;
      vec3f Ngff
        = hitWasOnFront
        ?   Ng
        : - Ng;

      if (hadNoIntersection) {
        // ==================================================================
        // regular ray that did NOT hit ANYTHING 
        // ==================================================================
        if (pathDepth == 0) {
          // ----------------------------------------------------------------
          // PRIMARY ray that didn't hit anything -> background
          // ----------------------------------------------------------------
          // fragment = path.missColor;
          fragment = primaryRayMissColor(world,renderer,path);
          // fragment = path.throughput * backgroundOrEnv(world,path);

          if (dbg)
            printf("miss primary %f %f %f -> %f %f %f\n",
                   path.missColor.x,
                   path.missColor.y,
                   path.missColor.z,
                   fragment.x,fragment.y,fragment.z);
        } else {
          // ----------------------------------------------------------------
          // SECONDARY ray that didn't hit anything -> env-light
          // ----------------------------------------------------------------
          // this path had at least one bounce, but now bounced into
          // nothingness - compute env-light contribution, and weigh it
          // with the path's carried throughput.
#if ENV_LIGHT_SAMPLING
# if USE_MIS
          const vec3f fromEnv = radianceFromEnv(world,renderer,path);
          fragment = (vec3f)path.throughput * fromEnv * (float)path.misWeight;

          if (dbg)
            printf("bounce ray hits env light: tp %f %f %f misweight %f fromEnv %f %f %f\n",
                   (float)path.throughput.x,
                   (float)path.throughput.y,
                   (float)path.throughput.z,
                   (float)path.misWeight,
                   fromEnv.x,
                   fromEnv.y,
                   fromEnv.z);
# else
          fragment
            = path.isSpecular
            ? radianceFromEnv(world,renderer,path)
            : vec3f(0.f);
# endif
#else
          const vec3f fromEnv = radianceFromEnv(world,renderer,path);
          if (dbg)
            printf("fromenv %f %f %f\n",
                   fromEnv.x,
                   fromEnv.y,
                   fromEnv.z);
          fragment = path.throughput * fromEnv;
#endif
        }
        // no outgoing rays; this path is done.
        path.tMax = -1.f;
        return;
      }
    

      // ==================================================================
      // this ray DID hit something: compute its local frame buffer
      // contribution at this hit point (if any), and generate secondary
      // ray and shadow ray (if applicable), with proper weights.
      // ==================================================================    
      Random &random = (Random &)path.rngSeed;
      const PackedBSDF bsdf = path.getBSDF();
      // bool doTransmission = false;
      // =  ((float)path.mini.transmission > 0.f)
      // && (random() < (float)path.mini.transmission);
      render::DG dg;
      dg.P  = path.P;
      dg.Ng = Ng;
      dg.Ns = Ng;
      dg.wo = -normalize((vec3f)path.dir);
      dg.insideMedium = path.isInMedium;

      // if the ray is a volume hit we want it offset it into the
      // direction the ray came from (otherwise we have a chance of
      // the shadow ray or boucne ray to terminate right where the
      // original ray ended; for others we want to offset based on
      // normal */
      const float offsetEpsilon = safe_eps(EPS,dg.P);
      vec3f frontFacingSurfaceOffset
        = (isVolumeHit?dg.wo:Ngff);

      // ==================================================================
      // FIRST, let us look at generating any shadow rays, if
      // applicable; this way we can later modify the incoming ray in
      // place when we generate the outgoing ray.
      // ==================================================================
      Light::Sample ls;
      // todo check if BSDF is perfectly specular
#if USE_MIS
      bool lightNeedsMIS = false;
      bool lightIsDirLight = false;
#endif
      if (dbg)
        printf("sampling lights with N %f %f %f\n",Ngff.x,Ngff.y,Ngff.z);
      if (sampleLights(ls,world,renderer,dg.P,Ngff,random,
#if USE_MIS
                       lightNeedsMIS,
                       lightIsDirLight,
#endif
                       dbg)) {
        if (dbg)
          printf("sample light dir %f %f %f rad %f %f %f pdf %f spike %f\n",
                 ls.direction.x,
                 ls.direction.y,
                 ls.direction.z,
                 ls.radiance.x,
                 ls.radiance.y,
                 ls.radiance.z,
                 ls.pdf,
                 reduce_max(ls.radiance)/ls.pdf);
        EvalRes f_r
          = bsdf.eval(dg,ls.direction,dbg);
        if (dbg) printf("eval light res %f %f %f: %f\n",
                        f_r.value.x,
                        f_r.value.y,
                        f_r.value.z,
                        f_r.pdf);
          
        if (!f_r.valid() || reduce_max(f_r.value) < 1e-4f) {
          if (dbg)
            printf(" no f_r, killing shadow ray\n");
          shadowRay.tMax = -1.f;
        } else {
#ifdef CLAMP_F_R
          f_r.value = min(f_r.value,vec3f(CLAMP_F_R));
#endif
          vec3f tp_sr
            = (incomingThroughput)
            * (1.f/ls.pdf)
            * f_r.value
            * ls.radiance
            // * ONE_OVER_PI
            * (isVolumeHit?1.f:fabsf(dot(dg.Ng,ls.direction)))
            ;

          
          if (dbg) {
            printf(" -> inc tp %f %f %f, dot %f\n",
                   incomingThroughput.x,
                   incomingThroughput.y,
                   incomingThroughput.z,
                   (isVolumeHit?1.f:fabsf(dot(dg.Ng,ls.direction))));
            printf(" -> shadow f_r %f %f %f ls.rad %f %f %f pdf %f\n",
                   f_r.value.x,
                   f_r.value.y,
                   f_r.value.z,
                   ls.radiance.x,
                   ls.radiance.y,
                   ls.radiance.z,
                   ls.pdf);
            printf(" -> shadow ray tp %f %f %f\n",
                   tp_sr.x,
                   tp_sr.y,
                   tp_sr.z);
          }
          shadowRay.makeShadowRay
            (/* thrghhpt */tp_sr,
             /* surface: */dg.P + offsetEpsilon*frontFacingSurfaceOffset,
             /* to light */ls.direction,
             /* length   */ls.distance * (1.f-2.f*offsetEpsilon));
          shadowRay.rngSeed = path.rngSeed + 1; random();
          shadowRay.dbg = path.dbg;
          shadowRay.pixelID = path.pixelID;
            
          shadowRay.misWeight = 1.f;
#if USE_MIS
          if (!lightIsDirLight && lightNeedsMIS) {
            float pdf_lightRay_lightDir
              = world.envMapLight.pdf(ls.direction);
            float pdf_scatterRay_lightDir
              = bsdf.pdf(dg,ls.direction);
            // compute MIS weight weight that shadow direction
            shadowRay.misWeight
              = pdf_lightRay_lightDir
              / (pdf_lightRay_lightDir + pdf_scatterRay_lightDir + 1e-10f);
            // and if it's too small for any reason, kill the shadow
            // ray
            if ((float)shadowRay.misWeight < 1e-5f)
              shadowRay.tMax  = -1.f;
          }
#endif
        }
      }
      
      // ==================================================================
      // now, let's decide what to do with the ray itself
      // ==================================================================

      // if we exceeded max depth we die, one way or another.
      if (pathDepth >= MAX_PATH_DEPTH) {
        path.tMax = -1.f;
        return;
      }
      // now per default, "create" a valid scatter ray.
      path.clearHit();
      path.tMax = BARNEY_INF;
      
      ScatterResult scatterResult;
      bsdf.scatter(scatterResult,dg,random,dbg);
#ifndef NDEBUG
      if (scatterResult.type == ScatterResult::INVALID)
        printf("broken BSDF, doesn't set scatter type!\n");
#endif 
      if (dbg)
        printf("scatter result.valid ? %i\n",
               int(scatterResult.valid()));
      if (!scatterResult.valid() || scatterResult.pdf <= 1e-6f)
        return;
      
      if (scatterResult.type == ScatterResult::VOLUME) {
#if SCI_VIS_MODE
        // sci vis mode: volumes do shadow, but nothing more
        path.tMax = -1.f;
        return;
#else
        // treat volume scatter like a diffuse scatter.
        scatterResult.type = ScatterResult::DIFFUSE;
#endif
      }
      
      if (scatterResult.type == ScatterResult::DIFFUSE ||
          scatterResult.type == ScatterResult::VOLUME) {
        if (path.numDiffuseBounces >= MAX_DIFFUSE_BOUNCES) {
          path.tMax = -1.f;
          return;
        } else
          path.numDiffuseBounces = path.numDiffuseBounces + 1;
      }
      path.isSpecular = (scatterResult.type == ScatterResult::SPECULAR);
      
      if (dbg)
        printf("offsetting into sign %f, direction %f %f %f\n",
               scatterResult.offsetDirection,
               frontFacingSurfaceOffset.x,
               frontFacingSurfaceOffset.y,
               frontFacingSurfaceOffset.z); 
      path.org
        = dg.P + scatterResult.offsetDirection * offsetEpsilon*frontFacingSurfaceOffset;
// #ifdef CLAMP_F_R
//       scatterResult.f_r = min(scatterResult.f_r,vec3f(100.f));
// #endif

      if (dbg)
        printf("path scattered, bsdf in scatter dir is %f %f %f, pdf %f\n",
               (float)scatterResult.f_r.x, 
               (float)scatterResult.f_r.y, 
               (float)scatterResult.f_r.z,
               scatterResult.pdf);
      path.dir
        = normalize(scatterResult.dir);
      
      vec3f scatterFactor
        = scatterResult.f_r
        // * (isVolumeHit?1.f:fabsf(dot(dg.Ng,path.dir)))
        // * ONE_OVER_PI
        / (isinf(scatterResult.pdf)? 1.f : (ONE_PI*scatterResult.pdf + 1e-10f));
      
#if 1
      // uhhhh.... this is TOTALLY wrong, but let's limit how much
      // each bounce can increase the throughput of a ray ... this
      // makes fireflies go away (well, makes them go 'less', but can
      // lose a lot of envergy if the brdf sample code isn't close to
      // the actual brdf.
      scatterFactor = min(scatterFactor,vec3f(1.5f));
#endif

      path.throughput
        = path.throughput * scatterFactor;
      if (dbg && scatterResult.changedMedium)
        printf("path DID change medium\n");
      if (scatterResult.changedMedium)
        path.isInMedium = !path.isInMedium;
      
      if (dbg)
        printf("scatter dir %f %f %f tp %f %f %f\n",
               (float)path.dir.x,
               (float)path.dir.y,
               (float)path.dir.z,
               (float)path.throughput.x,
               (float)path.throughput.y,
               (float)path.throughput.z);
      
      
#if USE_MIS
      if (lightNeedsMIS && !isinf(scatterResult.pdf)) {
        float pdf_scatterRay_scatterDir = bsdf.pdf(dg,path.dir);
        float pdf_lightRay_scatterDir   = world.envMapLight.pdf(path.dir);
        
        path.misWeight
          = pdf_scatterRay_scatterDir
          / (pdf_scatterRay_scatterDir + pdf_lightRay_scatterDir);
      } else {
        path.misWeight = 1.f;
      }
#endif
    }
#endif // device code  

    struct ShadeRaysKernel {
      inline __rtc_device
      void run(const rtc::ComputeInterface &rt);
      
      World::DD world;
      Renderer::DD renderer;
      AccumTile *accumTiles;
      int accumID;
      Ray *readQueue;
      int numRays;
      Ray *writeQueue;
      int *d_nextWritePos;
      int generation;
    };

#if RTC_DEVICE_CODE
    inline __rtc_device
    void ShadeRaysKernel::run(const rtc::ComputeInterface &rt)
    {
      int tid = rt.getThreadIdx().x + rt.getBlockIdx().x*rt.getBlockDim().x;
      if (tid >= numRays) return;

      Ray path = readQueue[tid];
#ifdef NDEBUG
      bool dbg = false;
#else
      bool dbg = path.dbg;
#endif

      /* note(iw): IMHO pixels that did _not_ hit any geometry should
         have an alpha value of 0, even if they did comptue a 'color'
         from the env map. However, TSD blends its renderd image over
         a black background, in which case that blends away background
         pixels, so for now turn that off. The "right" fix for this
         would be for TSD to handle alpha properly. */
#define COMPUTE_PROPER_ALPHA_CHANNEL 0
      float alpha 
        = (generation == 0)
        ?
#if COMPUTE_PROPER_ALPHA_CHANNEL
        (path.hadHit()? 1.f : path.missColor.w)
#else
        1.f
#endif
        : 0.f;
#if DENOISE
      vec3f incomingN
        = path.hadHit()
        ? path.getN()
        : vec3f(0.f);
      if (incomingN == vec3f(0.f))
        incomingN = vec3f(1.f,0.f,0.f);
#endif
      // what we'll add into the frame buffer
      vec3f fragment = 0.f;
      float z = path.tMax;
      // create a (potential) shadow ray, and init to 'invalid'
      Ray shadowRay;
      shadowRay.tMax = -1.f;
      
      // bounce that ray on the scene, possibly generating a) a fragment
      // to add to frame buffer; b) a outgoing ray (in-place
      // modification of 'path'); and/or c) a shadow ray
      bounce(world,renderer,
             fragment,
             path,shadowRay,
             generation);
    
      // write shadow and bounce ray(s), if any were generated
      if (dbg)
        printf("path.tmax %f shadowray.tmax %f frag %f %f %f\n",
               path.tMax,shadowRay.tMax,
               fragment.x,fragment.y,fragment.z);
      if (shadowRay.tMax > 0.f) {
        writeQueue[rt.atomicAdd(d_nextWritePos,1)] = shadowRay;
      }
      if (path.tMax > 0.f) {
        writeQueue[rt.atomicAdd(d_nextWritePos,1)] = path;
      }

      // and write the shade fragment, if generated
      int tileID  = int(path.pixelID / pixelsPerTile);
      int tileOfs = int(path.pixelID % pixelsPerTile);
      vec4f &valueToAccumInto
        = accumTiles[tileID].accum[tileOfs];

#if DENOISE
      vec3f &valueToAccumNormalInto
        = accumTiles[tileID].normal[tileOfs];
#endif
      
      // ==================================================================
      // add to accum buffer. be careful of two things:
      //
      // a) since each pixel could have two DIFFERENT rays in the
      // queue (shadow ray and bounce ray) we cannot simply 'add', but
      // have to use an atomic add, because these could be in the same
      // warp.
      //
      // b) since we don't have an explicit frame buffer clear we
      // still have to make sure each pixel is written - not added -
      // exactly once in the first generation of the first frame.
      // ==================================================================

      // clamping ...
      float clampMax = 10.f*(1+accumID);
      fragment = min(fragment,vec3f(clampMax));

      if (accumID == 0 && generation == 0) {
        valueToAccumInto = vec4f(fragment.x,fragment.y,
                                 fragment.z,alpha);
      } else {
        if (generation == 0 && alpha) 
          rt.atomicAdd(&valueToAccumInto.w,alpha);

        if (fragment.x > 0.f)
          rt.atomicAdd(&valueToAccumInto.x,fragment.x);
        if (fragment.y > 0.f)
          rt.atomicAdd(&valueToAccumInto.y,fragment.y);
        if (fragment.z > 0.f)
          rt.atomicAdd(&valueToAccumInto.z,fragment.z);
#if DENOISE
        if (incomingN.x > 0.f)
          rt.atomicAdd(&valueToAccumNormalInto.x,incomingN.x);
        if (incomingN.y > 0.f)
          rt.atomicAdd(&valueToAccumNormalInto.y,incomingN.y);
        if (incomingN.z > 0.f)
          rt.atomicAdd(&valueToAccumNormalInto.z,incomingN.z);
#endif
      }

      // and for apps that need a depth buffer, write z
      if (generation == 0) {
        float &tile_z = accumTiles[tileID].depth[tileOfs];
        if (accumID == 0) 
          tile_z = z;
        else
          tile_z = min(tile_z,z);
      }
    }
#endif
  }  

  using namespace render;
  
  void Context::shadeRaysLocally(Renderer *renderer,
                                 GlobalModel *model,
                                 FrameBuffer *fb,
                                 int generation)
  {
    for (auto slotModel : model->modelSlots) {
      World *world = slotModel->world.get();
      for (auto device : *world->devices) {
        SetActiveGPU forDuration(device);
        RayQueue *rayQueue = device->rayQueue;
        device->rayQueue->resetWriteQueue();
        
        TiledFB *devFB     = fb->getFor(device);
        int numRays        = rayQueue->numActive;
        if (numRays == 0) continue;
        int bs = 128;
        int nb = divRoundUp(numRays,bs);
        World::DD devWorld
          = world->getDD(device);
        Renderer::DD devRenderer
          = renderer->getDD(device);

        render::ShadeRaysKernel args = {
          devWorld,devRenderer,
          devFB->accumTiles,
          (int)fb->accumID,
          rayQueue->traceAndShadeReadQueue,
          numRays,
          rayQueue->receiveAndShadeWriteQueue,
          rayQueue->_d_nextWritePos,
          generation,
        };
        device->shadeRays->launch(nb,bs,&args);
      }
    }

    // ------------------------------------------------------------------
    // wait for kernel to complete, and swap queues 
    // ------------------------------------------------------------------
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      device->rtc->sync();
      device->rayQueue->swap();
      device->rayQueue->numActive = device->rayQueue->readNumActive();
    }
  }
  
  RTC_EXPORT_COMPUTE1D(shadeRays,BARNEY_NS::render::ShadeRaysKernel);
}

